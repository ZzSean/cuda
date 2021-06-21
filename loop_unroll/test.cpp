#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

void loop_unroll1(void) {
  float a[1000000];
  for (int i = 0; i < 1000000; i++)
    a[i] = a[i] + 3;
}

void loop_unroll2(void) {
  float a[1000000];
  for (int i = 0; i < 1000000; i += 2) {
    a[i] = a[i] + 3;
    a[i + 1] = a[i + 1] + 3;
  }
}

void loop_unroll3(void) {
  float a[1000000];
  for (int i = 0; i < 1000000; i += 4) {
    a[i] = a[i] + 3;
    a[i + 1] = a[i + 1] + 3;
    a[i + 2] = a[i + 2] + 3;
    a[i + 3] = a[i + 3] + 3;
  }
}

int main(int argc, char **argv) {
  struct timeval time_start, time_end;
  gettimeofday(&time_start, NULL);
  loop_unroll1();
  gettimeofday(&time_end, NULL);
  printf("loop_unroll_1 time: %ldus\n", time_end.tv_usec - time_start.tv_usec);
  gettimeofday(&time_start, NULL);
  loop_unroll1();
  gettimeofday(&time_end, NULL);
  printf("loop_unroll_1 time: %ldus\n", time_end.tv_usec - time_start.tv_usec);
  gettimeofday(&time_start, NULL);
  loop_unroll2();
  gettimeofday(&time_end, NULL);
  printf("loop_unroll_2 time: %ldus\n", time_end.tv_usec - time_start.tv_usec);
  gettimeofday(&time_start, NULL);
  loop_unroll3();
  gettimeofday(&time_end, NULL);
  printf("loop_unroll_4 time: %ldus\n", time_end.tv_usec - time_start.tv_usec);

  return 0;
}
