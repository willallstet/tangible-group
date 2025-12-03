  const int ledPins[] = {2, 3, 4, 5, 6, 7, 8};
const int ledCount = sizeof(ledPins) / sizeof(int);

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < ledCount; i++) {
    pinMode(ledPins[i], OUTPUT);
    digitalWrite(ledPins[i], LOW);
  }
}

void loop() {
  if (!Serial.available()) return;

  String line = Serial.readStringUntil('\n');
  line.trim();

  bool active[ledCount] = {false};

  if (line.length() > 0) {
    line += ",";
    int start = 0;
    while (start < line.length()) {
      int comma = line.indexOf(',', start);
      if (comma == -1) break;
      String token = line.substring(start, comma);
      token.trim();
      if (token.length() > 0) {
        int pos = token.toInt();
        if (pos >= 0 && pos < ledCount) {
          active[pos] = true;
        }
      }
      start = comma + 1;
    }
  }

  for (int i = 0; i < ledCount; i++) {
    digitalWrite(ledPins[i], active[i] ? HIGH : LOW);
  }
}