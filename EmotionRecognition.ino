#include <Servo.h>
Servo servo1;
Servo servo2;
int x_pos;
int y_pos;
int servo1_pin = 9;
int servo2_pin = 10;  
int initial_position = 0;
int initial_position1 = 180;
int val=-1;
void setup ( ) {
Serial.begin (9600) ;
servo1.attach (servo1_pin ) ; 
servo2.attach (servo2_pin ) ; 
servo1.write (initial_position);
servo2.write (initial_position1);            
}

void loop ( ) 
{
  val=-1;
if(Serial.available()>0)
{
  String incoming = Serial.readStringUntil('\n');
   val = incoming.toInt();  
}
if(val==1)
{
  servo1.write ( 180 ) ;
  delay (1000) ;
  servo1.write ( 0 ) ;
  delay (1000) ;
}
if(val==0)
{
  servo2.write ( 0 ) ;
  delay (1000) ;
  servo2.write ( 180 ) ;
  delay (1000) ;
}

}