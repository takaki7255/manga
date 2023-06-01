#!/bin/sh
set -e
if test "$CONFIGURATION" = "Debug"; then :
  cd /Users/tanakakaito/src/motioncomic/コマ分割精度向上/motioncomicGenerator
  make -f /Users/tanakakaito/src/motioncomic/コマ分割精度向上/motioncomicGenerator/CMakeScripts/ReRunCMake.make
fi
if test "$CONFIGURATION" = "Release"; then :
  cd /Users/tanakakaito/src/motioncomic/コマ分割精度向上/motioncomicGenerator
  make -f /Users/tanakakaito/src/motioncomic/コマ分割精度向上/motioncomicGenerator/CMakeScripts/ReRunCMake.make
fi
if test "$CONFIGURATION" = "MinSizeRel"; then :
  cd /Users/tanakakaito/src/motioncomic/コマ分割精度向上/motioncomicGenerator
  make -f /Users/tanakakaito/src/motioncomic/コマ分割精度向上/motioncomicGenerator/CMakeScripts/ReRunCMake.make
fi
if test "$CONFIGURATION" = "RelWithDebInfo"; then :
  cd /Users/tanakakaito/src/motioncomic/コマ分割精度向上/motioncomicGenerator
  make -f /Users/tanakakaito/src/motioncomic/コマ分割精度向上/motioncomicGenerator/CMakeScripts/ReRunCMake.make
fi

