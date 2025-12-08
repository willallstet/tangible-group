#include <Adafruit_NeoPixel.h>

#define PIXEL_PIN 3
#define NUMPIXELS 13

Adafruit_NeoPixel strip(NUMPIXELS, PIXEL_PIN, NEO_GRB + NEO_KHZ800);

// =====================================================
// STRUCT: Position → LEDs → Microswitch
// =====================================================

struct BookSpace {
  const char* id;       // e.g. "A1"
  uint8_t switchPin;    // microswitch pin (INPUT_PULLUP)
  uint8_t leds[4];      // LEDs belonging to this space
  uint8_t ledCount;     // number of LEDs
  bool lastState;       // for detecting state changes
};

// =====================================================
// DEFINE SPACES — TINKERCAD TEST VERSION
// =====================================================
//
// A1 → LEDs {0,1}, microswitch pin 3
// A2 → LED  {3},   microswitch pin 4
// A3 → LEDs {5,6}, microswitch pin 5

BookSpace spaces[] = {
  { "A1", 5, {0,1}, 2, false },
  { "A2", 7, {2},    1, false },
  { "A3", 8, {4}, 1, false },
  { "A4", 9, {6}, 1, false },
  { "A5", 10, {8,9},    2, false },
  { "A6", 12, {11,12}, 2, false },


  
};

const int NUM_SPACES = sizeof(spaces) / sizeof(spaces[0]);

// =====================================================
// Helpers
// =====================================================

// Convert hex colour "#RRGGBB" to R,G,B
void parseHexColor(String hex, int &r, int &g, int &b) {
  hex.trim();
  if (hex.startsWith("#")) hex.remove(0, 1);

  long number = strtol(hex.c_str(), NULL, 16);
  r = (number >> 16) & 0xFF;
  g = (number >> 8)  & 0xFF;
  b = number & 0xFF;
}

// Return index of BookSpace by ID string
int findSpaceIndex(String id) {
  id.trim();
  for (int i = 0; i < NUM_SPACES; i++) {
    if (id.equalsIgnoreCase(spaces[i].id)) return i;
  }
  return -1;
}

// =====================================================
// SERIAL COMMAND FORMAT:
//    A1,ON,#FF0000
//    A2,OFF
// =====================================================

void parseCommand(String cmd) {
  cmd.trim();

  // split on commas
  int firstComma = cmd.indexOf(',');
  if (firstComma < 0) return;

  String id = cmd.substring(0, firstComma);
  int idx = findSpaceIndex(id);
  if (idx < 0) {
    Serial.println("Unknown ID");
    return;
  }

  String rest = cmd.substring(firstComma + 1);
  rest.trim();

  // second comma (optional)
  int secondComma = rest.indexOf(',');

  String action = (secondComma == -1) ?
                  rest :
                  rest.substring(0, secondComma);

  action.trim();
  action.toUpperCase();

  // OFF = turn off LEDs
  if (action == "OFF") {
    for (int i = 0; i < spaces[idx].ledCount; i++) {
      strip.setPixelColor(spaces[idx].leds[i], 0, 0, 0);
    }
    strip.show();
    Serial.print(id); Serial.println(" LEDs OFF");
    return;
  }

  // ON + colour required
  if (action == "ON") {
    if (secondComma == -1) {
      Serial.println("ON requires colour");
      return;
    }

    String hex = rest.substring(secondComma + 1);
    int r, g, b;
    parseHexColor(hex, r, g, b);

    // set LEDs
    for (int i = 0; i < spaces[idx].ledCount; i++) {
      strip.setPixelColor(spaces[idx].leds[i], r, g, b);
    }
    strip.show();

    Serial.print(id);
    Serial.print(" LEDs → ");
    Serial.println(hex);
  }
}

// =====================================================
// SETUP
// =====================================================

void setup() {
  Serial.begin(9600);

  strip.begin();
  strip.clear();
  strip.show();

  // configure microswitches
  for (int i = 0; i < NUM_SPACES; i++) {
    pinMode(spaces[i].switchPin, INPUT_PULLUP);
    spaces[i].lastState = digitalRead(spaces[i].switchPin) == LOW;
  }

  Serial.println("Ready. Try: A1,ON,#FF0000");
}

// =====================================================
// LOOP
// =====================================================

void loop() {

  // ---- SERIAL COMMANDS ----
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    parseCommand(cmd);
  }

  // ---- BUTTON STATE CHECK ----
  for (int i = 0; i < NUM_SPACES; i++) {
    bool pressed = (digitalRead(spaces[i].switchPin) == LOW);

    if (pressed != spaces[i].lastState) {
      spaces[i].lastState = pressed;

      Serial.print(spaces[i].id);
      Serial.print(", ");
      Serial.println(pressed ? "PRESSED" : "RELEASED");
    }
  }
}