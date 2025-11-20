#include <ESP8266WiFi.h>
#include <FirebaseESP8266.h>
#include <DHT.h>
#include <WiFiUdp.h>
#include <NTPClient.h>

/* WiFi Credentials */
#define WIFI_SSID "your_wifi"
#define WIFI_PASS "your_pass"

/* Firebase Credentials */
#define FIREBASE_HOST "iot-forensics-e8c95-default-rtdb.asia-southeast1.firebasedatabase.app"
#define FIREBASE_AUTH "YOUR_FIREBASE_DATABASE_SECRET"

/* DHT Sensor */
#define DHTPIN 2
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

/* Firebase */
FirebaseData fbData;

/* Time Setup */
WiFiUDP ntpUDP;
NTPClient timeClient(ntpUDP, "pool.ntp.org", 19800);  // +5:30 (IST)

/* Firebase Path */
String firebasePath = "/forensics_logs";

void setup() {
  Serial.begin(115200);

  dht.begin();

  Serial.println("Connecting to WiFi...");
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi Connected!");
  Serial.println("IP Address: " + WiFi.localIP().toString());

  Firebase.begin(FIREBASE_HOST, FIREBASE_AUTH);

  timeClient.begin();
  timeClient.update();
}

void loop() {

  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();

  if (isnan(temperature) || isnan(humidity)) {
    Serial.println("Failed to read from DHT sensor!");
    delay(2000);
    return;
  }

  timeClient.update();
  long timestamp = timeClient.getEpochTime();

  Serial.println("Sending data to Firebase...");

  FirebaseJson json;
  json.add("temperature", temperature);
  json.add("humidity", humidity);
  json.add("anomaly", "normal");
  json.add("timestamp", timestamp);

  if (Firebase.pushJSON(fbData, firebasePath, json)) {
    Serial.println("Data pushed successfully!");
    Serial.println(fbData.pushName());
  } else {
    Serial.println("Failed to push data");
    Serial.println(fbData.errorReason());
  }

  delay(5000);  // Send every 5 seconds
}
