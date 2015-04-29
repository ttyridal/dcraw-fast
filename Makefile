ARCH:=-march=native
#-msse4 -msse3 -msse
#ARCH:=$(ARCH) -fprofile-generate
#ARCH:=$(ARCH) -fprofile-use

OPTIM:=-O4 -ffast-math  -ftree-vectorize

LIBS:=-lm -pthread

DEFINES:=-DTHREADED

IGNORE_WARNINGS=-Wno-clobbered \
				-Wno-unused-result \
				-Wno-array-bounds \
				-Wno-maybe-uninitialized \
				-Wno-aggressive-loop-optimizations \
				-Wno-sign-compare \
				-Wno-missing-field-initializers





dcraw: dcraw.o dcraw_ahdfast.o convert_rgb.o
	gcc $(ARCH) -Wall -Wextra -o $@ $^ $(LIBS) $(OPTIM)
dcraw.o: dcraw.c
	@gcc -c $(ARCH) -Wall -Wextra $(IGNORE_WARNINGS) $(DEFINES) -DNODEPS $(OPTIM) $<

dcraw_ahdfast.o: dcraw_ahdfast.c
	@gcc -c -std=gnu99  $(ARCH) $(DEFINES) -Werror -Wall -Wextra $(OPTIM) $<
ljpeg_fast.o: ljpeg_fast.c
	@gcc -c -std=gnu99 $(ARCH) $(DEFINES) -Werror -Wall -Wextra $(OPTIM) $<
convert_rgb.o: convert_rgb.c
	@gcc -c -std=gnu99 $(ARCH) $(DEFINES) -Werror -Wall -Wextra $(OPTIM) $<

.PHONY: test
test: dcraw
	./dcraw -w -6 -v test.cr2
	diff test_golden.ppm test.ppm

.PHONY: test1
test1: 
	gcc $(ARCH) -o test test.c && ./test

.PHONY: clean gcov profileassisted profileassisted1 profilepost profilepost1 profilepost2
clean:
	rm -f *.o dcraw *.gcda *.gcno gmon.out

profileassisted1: OPTIM:=$(OPTIM) -fprofile-generate
profileassisted1: clean dcraw
	./dcraw -w -6 -v test.cr2
profileassisted: profilepost

profilepost1: profileassisted1
	@rm -f *.o dcraw

profilepost: profileassisted1 profilepost1
	make profilepost2
profilepost2: OPTIM:=$(OPTIM) -fprofile-use
profilepost2: dcraw


gcov: OPTIM:=-g -O0 -fprofile-arcs -ftest-coverage  -ffast-math  -ftree-vectorize -pg
gcov: clean dcraw
	rm -Rf cov
	rm -f coverage.info
	./dcraw -w -6 -v test.cr2
	lcov --capture --directory . --output-file coverage.info
	mkdir cov
	genhtml coverage.info --output-directory cov

.PHONY: callgrind
callgrind:
	rm -f callgrind.out.*
	valgrind --tool=callgrind ./dcraw -w -6 -v test.cr2
