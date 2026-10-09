CC ?= cc
PKG_CONFIG ?= pkg-config
CFLAGS ?= -O3 -std=c11 -Wall -Wextra
FFMPEG_PACKAGES = libavformat libavcodec libavutil libswscale

.PHONY: all check clean
all: extract_codec_signals

extract_codec_signals: extract_codec_signals.c
	$(PKG_CONFIG) --exists $(FFMPEG_PACKAGES)
	$(CC) $(CFLAGS) $(CPPFLAGS) $$($(PKG_CONFIG) --cflags $(FFMPEG_PACKAGES)) $< -o $@ $(LDFLAGS) $$($(PKG_CONFIG) --libs $(FFMPEG_PACKAGES))

check: extract_codec_signals
	python3 validate_codecs.py --extractor ./extract_codec_signals

clean:
	rm -f extract_codec_signals
