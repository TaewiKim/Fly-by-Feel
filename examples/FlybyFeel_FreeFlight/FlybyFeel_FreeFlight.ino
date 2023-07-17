#include <string.h>

unsigned long timer = 0;
long loopTime = 4;   // microseconds
double deltaT = 500;   // us    minimum of 500 us, but set >1000us to be safe

int motorPin_front = 5;
int motorPin_tail = 22;
int motorCmd_front = 0;
int motorCmd_tail = 0;
int i = 0;

void setup() {
  Serial.begin(19200);
  pinMode(motorPin_front, OUTPUT);
  pinMode(3, OUTPUT);
  pinMode(4, OUTPUT);
  pinMode(motorPin_tail, OUTPUT);
  pinMode(20, OUTPUT);
  pinMode(21, OUTPUT);
  digitalWrite(3, HIGH);
  digitalWrite(4, LOW);  
}

void loop() {

  timeSync(loopTime);
  getSerialData();
  analogWrite(motorPin_front, motorCmd_front);
  analogWrite(motorPin_tail, motorCmd_tail);
  
  i = i+1;
  if (i > 1000000)
  {  
    motorCmd_front = 0;
    motorCmd_tail = 0;
    i = 0;
  }

}



void timeSync(unsigned long deltaT) {
  unsigned long currTime = micros();
  long timeToDelay = deltaT - (currTime - timer);
  if (timeToDelay > 5000)
  {
    delay(timeToDelay / 1000);
    delayMicroseconds(timeToDelay % 1000);
  }
  else if (timeToDelay > 0)
  {
    delayMicroseconds(timeToDelay);
  }
  else
  {
    // timeToDelay is negative so we start immediately
  }
  timer = currTime + timeToDelay;
}


void getSerialData()
{
  while (Serial.available())
  {
    char input = (char)Serial.read();
    String tmp = "";
    switch (input)
    {
      case 'T':
        tmp = getVal();
        if (tmp != "X")
        {
          if (tmp.toInt() > 250)
          {
            motorCmd_front = 250;
          }
          else
          {
            motorCmd_front = tmp.toInt();
          }
        }
        break;

      case 'D':
        tmp = getVal();
        if (tmp != "X")
        {
          if (tmp.toInt() > 0)
          {
            digitalWrite(20, LOW);
            digitalWrite(21, HIGH);
            if (tmp.toInt() > 120)
              motorCmd_tail = 120;
            else
              motorCmd_tail = tmp.toInt();
          }
          else if (tmp.toInt() < 0)
          {
            digitalWrite(21, LOW);
            digitalWrite(20, HIGH);
            if (tmp.toInt() < -120)
              motorCmd_tail = -120;
            else
              motorCmd_tail = -tmp.toInt();
          }
          else
          {
            motorCmd_tail = 0;
          }
        }
        break;

    }
  }
}

String getVal()
{
  String recvString = "";
  while (Serial.available())
  {
    char input = Serial.read();
    if (input == '%')   // this is the end of message marker so that the program knows when to update the g_scaleFactor variable
    {
      return recvString;
    }
    recvString += input;
  }

  return "X";   // failed to receive the EOM marker
}
