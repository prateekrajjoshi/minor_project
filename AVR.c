/*
 * communication.c
 *
 * Created: 8/10/2015 2:20:19 PM
 *  Author: prateek
 */ 

//FOR ATMEGA8 AS TX WHEN SIGNAL IS SENT

#define F_CPU 12000000
#include <avr/io.h>
#include <util/delay.h>


/*#include "uart.cpp"
#define baudRate 9600
#define xtalCpu 12000000l
#define UART_BAUD_SELECT(baudRate, xtalCpu) ((xtalCpu)/((baudRate)*16l)-1) */

int main(void)
{
	
	
	char forward = 'a';
	char backward = 'b';
	char turn_left ='c';
	char turn_right = 'd';
	char stop = 'e';
	
	
	int UBBR_Value = 77;                                 //baud rate of 2400
	UBRRH = (unsigned char)(UBBR_Value>>8);
	UBRRL = (unsigned char)UBBR_Value;
	
	UCSRB|= (1<<RXEN)|(1<<TXEN);                           //tx and rx enable 
	
	UCSRC |= (0<<USBS)|(1 << URSEL)|(3<<UCSZ0);                      // 1 stop bits | selectt| 8 bits data
	
	
	DDRD = 0xff;
	DDRB = 0xff;
	unsigned char rxdata;
	PORTD=0b01000000;
	while(1){
		
			
			while (!( UCSRA & (1<<RXC)));    //while 0, polling udre bit in ucsra register
				rxdata =(unsigned char)  UDR;									
			
			if(rxdata==forward) {
				
				PORTD ^= 0b11000000;
				PORTB=0b00001001;
			}
			else if(rxdata==backward) {
				PORTD ^= 0b11000000;
				PORTB=0b00000110;
			}
			else if(rxdata==turn_left) {
				PORTD ^= 0b11000000;
				PORTB=0b00000101;
				
			}
			else if(rxdata==turn_right) {
				PORTD ^= 0b11000000;
				PORTB=0b00001010;
			}
			
			else          // if any unrecognized stop
			{ PORTD= 0b10000000;
				PORTB=0;
			
			}				
		}
		
		
		
		
	}		
   
  

