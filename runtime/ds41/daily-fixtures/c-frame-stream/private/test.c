#include "frame_parser.h"
#include <assert.h>
#include <string.h>
int main(void){ FrameParser p; char out[4][128]; size_t n=0; frame_parser_init(&p);
 assert(frame_parser_feed(&p,(const unsigned char*)"alp",3,out,4,&n)==0 && n==0);
 assert(frame_parser_feed(&p,(const unsigned char*)"ha\r\nbeta",8,out,4,&n)==0 && n==1 && strcmp(out[0],"alpha")==0);
 assert(frame_parser_feed(&p,(const unsigned char*)"-two\n",5,out,4,&n)==0 && n==1 && strcmp(out[0],"beta-two")==0);
 unsigned char big[140]; memset(big,'x',sizeof(big)); assert(frame_parser_feed(&p,big,sizeof(big),out,4,&n)==0); assert(frame_parser_feed(&p,(const unsigned char*)"\nok\n",4,out,4,&n)==0 && n==1 && strcmp(out[0],"ok")==0); return 0; }
