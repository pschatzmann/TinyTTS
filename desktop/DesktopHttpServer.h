#pragma once
// Minimal single-threaded HTTP/1.1 server for `tinytts --serve PORT` -- the
// desktop counterpart of examples/tts_webserver (same endpoints):
//
//   GET  /tts?text=Hello%20world        -> audio/mpeg or audio/wav (streamed)
//   POST /tts   body: Hello world       -> (any Content-Type but form data)
//   POST /tts   body: text=Hello+world  -> (application/x-www-form-urlencoded)
//   GET  /?text=Hello%20world           -> the same as /tts
//   GET  /                              -> a small HTML form
//
// Plain POSIX sockets: under IS_MIN_DESKTOP arduino-audio-tools has no
// WiFiServer/Client emulation to build on. One request at a time (the
// model isn't reentrant anyway), every reply closes the connection -- which
// also marks the end of the audio, since its length isn't known up front.
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>

namespace tinytts {

/**
 * @brief Serves text-to-speech requests over HTTP: the text is extracted
 * from the request and handed to the synthesize callback, which streams the
 * reply body (a WAV file) to the client while it is being synthesized.
 * @author Phil Schatzmann
 * @copyright Apache-2.0
 */
class DesktopHttpServer {
 public:
  /// Sends reply data to the client; false once the client has gone away
  using WriteFn = std::function<bool(const char* data, size_t len)>;
  /// Writes the encoded audio for `text` via `write`
  using SynthesizeFn = std::function<bool(const std::string& text, const WriteFn& write)>;

  DesktopHttpServer(SynthesizeFn fn, const char* mime = "audio/wav") : synthesize_(std::move(fn)), mime_(mime) {}

  ~DesktopHttpServer() {
    if (fd_ >= 0) ::close(fd_);
  }

  bool begin(int port) {
    fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd_ < 0) {
      std::perror("socket");
      return false;
    }
    int yes = 1;
    ::setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons((uint16_t)port);
    if (::bind(fd_, (sockaddr*)&addr, sizeof(addr)) < 0 || ::listen(fd_, 8) < 0) {
      std::perror("bind/listen");
      return false;
    }
    std::fprintf(stderr, "Serving on http://0.0.0.0:%d/ (GET/POST /tts)\n", port);
    return true;
  }

  /// Handles requests until the process is terminated
  void run() {
    while (true) {
      int client = ::accept(fd_, nullptr, nullptr);
      if (client < 0) continue;
      // don't let a stalled client block the server forever
      timeval tv{10, 0};
      ::setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
      processClient(client);
      ::close(client);
    }
  }

 private:
  static constexpr size_t kMaxHeaderBytes = 16 * 1024;
  static constexpr size_t kMaxBodyBytes = 1024 * 1024;

  SynthesizeFn synthesize_;
  const char* mime_;
  int fd_ = -1;

  void processClient(int client) {
    // read up to the end of the header
    std::string data;
    size_t header_end;
    char buf[4096];
    while ((header_end = data.find("\r\n\r\n")) == std::string::npos) {
      if (data.size() > kMaxHeaderBytes) return sendText(client, 431, "Request Header Fields Too Large", "header too large\n");
      ssize_t n = ::recv(client, buf, sizeof(buf), 0);
      if (n <= 0) return;
      data.append(buf, n);
    }
    std::string header = data.substr(0, header_end);
    std::string body = data.substr(header_end + 4);

    // request line: METHOD SP PATH SP PROTOCOL
    size_t eol = header.find("\r\n");
    std::string line = header.substr(0, eol);
    size_t sp1 = line.find(' ');
    size_t sp2 = line.find(' ', sp1 + 1);
    if (sp1 == std::string::npos || sp2 == std::string::npos)
      return sendText(client, 400, "Bad Request", "bad request\n");
    std::string method = line.substr(0, sp1);
    std::string path = line.substr(sp1 + 1, sp2 - sp1 - 1);
    std::string query;
    size_t q = path.find('?');
    if (q != std::string::npos) {
      query = path.substr(q + 1);
      path = path.substr(0, q);
    }
    std::fprintf(stderr, "%s %s\n", method.c_str(), path.c_str());

    // "/?text=..." speaks as well: easy to mistype in the browser's address bar
    if ((path == "/tts" || (path == "/" && !getParameter(query, "text").empty())) && method == "GET") {
      speak(client, getParameter(query, "text"));
    } else if (path == "/tts" && method == "POST") {
      size_t len = std::strtoul(headerValue(header, "Content-Length").c_str(), nullptr, 10);
      if (len > kMaxBodyBytes) return sendText(client, 413, "Payload Too Large", "text too long\n");
      while (body.size() < len) {
        ssize_t n = ::recv(client, buf, sizeof(buf), 0);
        if (n <= 0) break;
        body.append(buf, n);
      }
      body.resize(std::min(body.size(), len));
      // form data (e.g. from the HTML form) -- but curl's --data also sends
      // plain text with this content type, so use the raw body if there is
      // no text field
      if (headerValue(header, "Content-Type").rfind("application/x-www-form-urlencoded", 0) == 0) {
        std::string text = getParameter(body, "text");
        if (!text.empty()) body = text;
      }
      speak(client, body);
    } else if (path == "/" && method == "GET") {
      sendReply(client, 200, "OK", "text/html", kHtmlForm);
    } else {
      sendText(client, 404, "Not Found", "not found\n");
    }
  }

  void speak(int client, const std::string& text) {
    if (text.find_first_not_of(" \t\r\n") == std::string::npos)
      return sendText(client, 400, "Bad Request", "missing text\n");
    // Streamed, sentence by sentence: the client can start playing early,
    // and a long text doesn't have to fit into memory (the Milk-V Duo has
    // ~100MB free). The status is sent first, so a failure later on can
    // only end the reply early.
    char header[160];
    int n = std::snprintf(header, sizeof(header), "HTTP/1.1 200 OK\r\nContent-Type: %s\r\nConnection: close\r\n\r\n",
                          mime_);
    if (!sendAll(client, header, n)) return;
    if (!synthesize_(text, [client](const char* data, size_t len) { return sendAll(client, data, len); }))
      std::fprintf(stderr, "synthesis aborted\n");
  }

  void sendText(int client, int code, const char* msg, const char* text) {
    sendReply(client, code, msg, "text/plain", text);
  }

  void sendReply(int client, int code, const char* msg, const char* type, const std::string& content) {
    char header[256];
    int n = std::snprintf(header, sizeof(header),
                          "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
                          "Connection: close\r\n\r\n",
                          code, msg, type, content.size());
    sendAll(client, header, n);
    sendAll(client, content.data(), content.size());
  }

  static bool sendAll(int client, const char* data, size_t len) {
    while (len > 0) {
      // MSG_NOSIGNAL: a client that hung up must not kill us with SIGPIPE
      ssize_t n = ::send(client, data, len, MSG_NOSIGNAL);
      if (n <= 0) return false;
      data += n;
      len -= n;
    }
    return true;
  }

  /// Case-insensitive lookup of a header field ("" if missing)
  static std::string headerValue(const std::string& header, const char* key) {
    size_t key_len = std::strlen(key);
    size_t pos = header.find("\r\n");
    while (pos != std::string::npos) {
      pos += 2;
      size_t end = header.find("\r\n", pos);
      std::string line = header.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
      if (line.size() > key_len && line[key_len] == ':' && strncasecmp(line.c_str(), key, key_len) == 0) {
        size_t b = line.find_first_not_of(' ', key_len + 1);
        return b == std::string::npos ? "" : line.substr(b);
      }
      pos = end;
    }
    return "";
  }

  /// Decodes %XX escapes and '+' (form encoding)
  static std::string urlDecode(const std::string& str) {
    std::string result;
    result.reserve(str.size());
    for (size_t i = 0; i < str.size(); i++) {
      char c = str[i];
      if (c == '+') {
        result += ' ';
      } else if (c == '%' && i + 2 < str.size() && std::isxdigit((unsigned char)str[i + 1]) &&
                 std::isxdigit((unsigned char)str[i + 2])) {
        result += (char)std::strtol(str.substr(i + 1, 2).c_str(), nullptr, 16);
        i += 2;
      } else {
        result += c;
      }
    }
    return result;
  }

  /// Extracts the value of `name` from "a=1&b=2" style data
  static std::string getParameter(const std::string& data, const char* name) {
    std::string key = std::string(name) + "=";
    size_t pos = 0;
    while (pos < data.size()) {
      size_t end = data.find('&', pos);
      if (end == std::string::npos) end = data.size();
      if (data.compare(pos, key.size(), key) == 0) return urlDecode(data.substr(pos + key.size(), end - pos - key.size()));
      pos = end + 1;
    }
    return "";
  }

  // One fetch() POST, played from the downloaded result: a plain <form> POST
  // makes the browser navigate to the audio, and its media player then drops
  // that connection and requests the URL again as GET -- without the text.
  // A short silent clip is started during the click: browsers that only allow
  // sound right after a user action (e.g. Firefox, Safari) then also play the
  // result, which arrives seconds later.
  static constexpr const char* kHtmlForm =
      "<!DOCTYPE html><html><head><meta charset='utf-8'>"
      "<meta name='viewport' content='width=device-width,initial-scale=1'>"
      "<title>TinyTTS</title></head><body>"
      "<h1>TinyTTS</h1>"
      "<textarea id='t' rows='4' cols='40'>Hello world!</textarea><br>"
      "<button id='b'>Speak</button> <span id='s'></span><br>"
      "<audio id='a' controls></audio>"
      "<script>"
      "b.onclick=async()=>{b.disabled=true;s.textContent='synthesizing...';"
      "a.src='data:audio/wav;base64,UklGRogAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YWQAAACAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICA';"
      "a.play().catch(()=>{});"
      "try{const r=await fetch('/tts',{method:'POST',"
      "headers:{'Content-Type':'text/plain'},body:t.value});"
      "if(!r.ok)throw new Error(await r.text());"
      "a.src=URL.createObjectURL(await r.blob());s.textContent='';"
      "a.play().catch(()=>{s.textContent='press play'});"
      "}catch(e){s.textContent='error: '+e.message}"
      "b.disabled=false};"
      "</script></body></html>";
};

}  // namespace tinytts
