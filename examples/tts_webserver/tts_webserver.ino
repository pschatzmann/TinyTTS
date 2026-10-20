/**
 * TinyTTS as a HTTP text-to-speech service (arduino-audio-tools) on ESP32-S3.
 *
 * Based on arduino-audio-tools' streams-generator-webserver_mp3 example, but
 * instead of streaming one fixed source the text to speak comes from the
 * request:
 *
 *   GET  /tts?text=Hello%20world        -> audio of "Hello world"
 *   POST /tts   body: Hello world       -> (Content-Type: text/plain)
 *   POST /tts   body: text=Hello+world  -> (application/x-www-form-urlencoded)
 *   GET  /?text=Hello%20world           -> the same as /tts
 *   GET  /                              -> a small HTML form for the browser
 *
 * e.g.
 *   curl "http://<ip>/tts?text=Hello%20world" -o hello.mp3
 *   curl -X POST --data-binary "Hello world" http://<ip>/tts -o hello.mp3
 *
 * The request is parsed with the audio-tools HttpRequestHeader and the PCM
 * that TinyTTS produces is encoded by an EncodedAudioStream which writes
 * directly to the WiFiClient: MP3 via MP3EncoderShine (requires
 * https://github.com/pschatzmann/codec-shine), or WAV if you comment out
 * USE_MP3 below.
 *
 * Synthesis on an ESP32 is far from real time (see docs/limitations.md), so
 * expect the reply to take a few seconds per second of speech -- the client
 * just waits. Only one request is handled at a time. A GET request line is
 * limited to HTTP_MAX_LEN (1024) bytes: use POST for longer texts.
 *
 * The model data is compiled into flash like in examples/tts_i2s_output, so
 * the same board settings and custom partitions.csv apply: PSRAM=opi,
 * FlashSize=16M, PartitionScheme=custom.
 */
#include <WiFi.h>
#include "TinyTTS.h"
#include "TinyTTS/data/default_weights_data.h"
#include "TinyTTS/data/default_cmudict_slim_data.h"
#include "TinyTTS/data/default_dictionary_model_data.h"

#include "AudioTools.h"
#include "AudioTools/Communication/AudioHttp.h"

#define USE_MP3
#ifdef USE_MP3
#include "AudioTools/AudioCodecs/CodecMP3Shine.h"
MP3EncoderShine encoder;
const char* mime = "audio/mpeg";
#else
WAVEncoder encoder;
const char* mime = "audio/wav";
#endif

// WIFI
const char* ssid = "ssid";
const char* password = "password";

const size_t kMaxTextLen = 4096;  // limit for POST bodies

WiFiServer server(80);
EncodedAudioStream encoded(&encoder);  // output defined per request
TinyTTS tts(encoded);
AudioInfo info;

// One fetch() POST, played from the downloaded result: a plain <form> POST
// makes the browser navigate to the audio, and its media player then drops
// that connection and requests the URL again as GET -- without the text.
// A short silent clip is started during the click: browsers that only allow
// sound right after a user action (e.g. Firefox, Safari) then also play the
// result, which arrives seconds later.
const char* html_form =
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

// Decodes %XX escapes and '+' (form encoding) in place
void urlDecode(std::string& str) {
  std::string result;
  result.reserve(str.size());
  for (size_t i = 0; i < str.size(); i++) {
    char c = str[i];
    if (c == '+') {
      result += ' ';
    } else if (c == '%' && i + 2 < str.size() && isxdigit(str[i + 1]) &&
               isxdigit(str[i + 2])) {
      result += (char)strtol(str.substr(i + 1, 2).c_str(), nullptr, 16);
      i += 2;
    } else {
      result += c;
    }
  }
  str = result;
}

// Extracts the value of `name` from "a=1&b=2" style data
std::string getParameter(const std::string& data, const char* name) {
  std::string key = std::string(name) + "=";
  size_t pos = 0;
  while (pos < data.size()) {
    size_t end = data.find('&', pos);
    if (end == std::string::npos) end = data.size();
    if (data.compare(pos, key.size(), key) == 0) {
      std::string value = data.substr(pos + key.size(), end - pos - key.size());
      urlDecode(value);
      return value;
    }
    pos = end + 1;
  }
  return "";
}

// Reads the request body (Content-Length bytes)
std::string readBody(WiFiClient& client, HttpRequestHeader& request) {
  std::string body;
  const char* len_str = request.get(CONTENT_LENGTH);
  size_t len = len_str == nullptr ? 0 : atoi(len_str);
  if (len > kMaxTextLen) len = kMaxTextLen;
  body.reserve(len);
  uint32_t timeout = millis() + 5000;
  while (body.size() < len && client.connected() && millis() < timeout) {
    int c = client.read();
    if (c >= 0) body += (char)c;
    else delay(1);
  }
  return body;
}

void sendReply(WiFiClient& client, int code, const char* msg,
               const char* contentType, const char* content = nullptr) {
  HttpReplyHeader reply;
  reply.setValues(code, msg, "HTTP/1.1");
  reply.put(CONTENT_TYPE, contentType);
  reply.put("Connection", "close");
  reply.write(client);
  if (content != nullptr) client.print(content);
}

void speak(WiFiClient& client, const std::string& text) {
  if (text.empty()) {
    sendReply(client, 400, "Bad Request", "text/plain", "missing text\n");
    return;
  }
  Serial.printf("speaking: %s\n", text.c_str());
  sendReply(client, 200, "OK", mime);
  // no Content-Length: the end of the data is signaled by closing the
  // connection
  encoded.setOutput(client);
  encoded.begin(info);
  uint32_t t0 = millis();
  tts.speak(text);
  encoded.end();
  Serial.printf("-> %lu ms\n", millis() - t0);
}

void processClient(WiFiClient& client) {
  HttpRequestHeader request;
  if (!request.read(client)) return;

  std::string path = request.urlPath();
  std::string query;
  size_t q = path.find('?');
  if (q != std::string::npos) {
    query = path.substr(q + 1);
    path = path.substr(0, q);
  }

  // "/?text=..." speaks as well: easy to mistype in the browser's address bar
  bool has_text = !getParameter(query, "text").empty();
  if ((path == "/tts" || (path == "/" && has_text)) && request.method() == GET) {
    speak(client, getParameter(query, "text"));
  } else if (path == "/tts" && request.method() == POST) {
    std::string body = readBody(client, request);
    const char* type = request.get(CONTENT_TYPE);
    // form data (e.g. from the HTML form) -- but curl's --data also sends
    // plain text with this content type, so use the raw body if there is no
    // text field
    if (type != nullptr &&
        StrView(type).startsWith("application/x-www-form-urlencoded")) {
      std::string text = getParameter(body, "text");
      if (!text.empty()) body = text;
    }
    speak(client, body);
  } else if (path == "/" && request.method() == GET) {
    sendReply(client, 200, "OK", "text/html", html_form);
  } else {
    sendReply(client, 404, "Not Found", "text/plain", "not found\n");
  }
}

void setup() {
  Serial.begin(115200);
  AudioToolsLogger.begin(Serial, AudioToolsLogLevel::Warning);

  // setup TTS
  tts.setWeights(default_weights, default_weights_len);
  tts.setDictionary(default_cmudict_slim, default_cmudict_slim_len);
  tts.setDictionaryModel(default_dictionary_model, default_dictionary_model_len);
  // full level: the audio is not played by a local amplifier
  tts.setVolume(1.0f);
  if (!tts.begin()) {
    Serial.println("TinyTTS.begin() failed -- did you call all setters above?");
    while (true) {}
  }
  // after tts.begin(): the sample rate is read from the model weights
  info = AudioInfo(tts.getAudioSampleRate(), tts.getAudioChannels(),
                   tts.getAudioBitsPerSample());

  // connect to WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(500);
  }
  WiFi.setSleep(false);
  Serial.println();
  Serial.print("Open http://");
  Serial.print(WiFi.localIP());
  Serial.println("/");

  // start server
  server.begin();
}

void loop() {
  WiFiClient client = server.accept();
  if (client) {
    processClient(client);
    client.flush();
    client.stop();
  }
}
