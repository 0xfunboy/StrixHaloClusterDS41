#include "ring.h"
#include <assert.h>
#include <string.h>
int main(void){unsigned char mem[8],out[16];Ring r;ring_init(&r,mem,8);assert(ring_write(&r,(unsigned char*)"abcdef",6)==6);assert(ring_read(&r,out,5)==5&&memcmp(out,"abcde",5)==0);assert(ring_write(&r,(unsigned char*)"1234567",7)==7);assert(ring_read(&r,out,8)==8&&memcmp(out,"f1234567",8)==0);ring_init(&r,mem,8);assert(ring_write(&r,(unsigned char*)"ABCDEFGH",8)==8);assert(ring_read(&r,out,8)==8&&memcmp(out,"ABCDEFGH",8)==0);return 0;}
