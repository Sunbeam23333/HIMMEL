/*
 * HIMMEL multi-codec motion/residual extractor.
 *
 * The extractor decodes any codec supported by libavcodec. It first requests
 * AV_FRAME_DATA_MOTION_VECTORS. When a decoder exports that side data (e.g.
 * H.264 and MPEG-4 Part 2 in stock FFmpeg), those vectors are written with
 * origin=bitstream. For decoders that do not export vectors (currently HEVC,
 * VP9 and AV1 in stock FFmpeg), a deterministic block-matching fallback is
 * used and written with origin=estimated.
 *
 * Residuals are diagnostic decoded-luma differences after an integer-coordinate
 * warp of the previous displayed frame. Native MVs can refer to a different
 * past frame; therefore this is NOT the exact codec predictor/residual, even
 * when origin=bitstream. Fractional MV values are preserved in the CSV, not in
 * this diagnostic warp. See docs/codec_extraction.md.
 *
 * Outputs:
 *   summary.csv
 *   motion_vectors/frame_XXXXXX.csv
 *   residuals/frame_XXXXXX.pgm
 *   residuals/frame_XXXXXX.s16le
 *
 * SPDX-License-Identifier: MIT
 * See LICENSE-MIT for the retained FFmpeg example notices.
 */

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/stat.h>
#include <sys/types.h>

#include <libavcodec/avcodec.h>
#include <libavcodec/version.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libavutil/motion_vector.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

typedef struct MotionBlock {
    int source;
    int w;
    int h;
    int src_x;
    int src_y;
    int dst_x;
    int dst_y;
    int motion_x;
    int motion_y;
    int motion_scale;
    uint64_t flags;
} MotionBlock;

typedef struct MotionList {
    MotionBlock *items;
    int count;
    int capacity;
    const char *origin;
} MotionList;

typedef struct Extractor {
    AVFormatContext *format;
    AVCodecContext *decoder;
    AVFrame *frame;
    AVPacket *packet;
    struct SwsContext *sws;
    int video_stream;
    int width;
    int height;
    uint8_t *current_gray;
    uint8_t *previous_gray;
    uint8_t *prediction;
    uint8_t *residual_vis;
    int16_t *residual_raw;
    int has_previous;
    int frame_number;
    int max_frames;
    int block_size;
    int search_radius;
    char output_dir[PATH_MAX];
    FILE *summary;
} Extractor;

static void print_av_error(const char *message, int err)
{
    char buffer[AV_ERROR_MAX_STRING_SIZE] = {0};
    av_strerror(err, buffer, sizeof(buffer));
    fprintf(stderr, "%s: %s\n", message, buffer);
}

static int mkdir_if_needed(const char *path)
{
    if (mkdir(path, 0755) == 0 || errno == EEXIST)
        return 0;
    fprintf(stderr, "Could not create directory %s: %s\n", path, strerror(errno));
    return -1;
}

static int prepare_output_directories(Extractor *ex)
{
    char path[PATH_MAX + 80];
    /* Refuse reuse: otherwise shorter reruns leave stale frame files. */
    if (mkdir(ex->output_dir, 0755) < 0) {
        fprintf(stderr, "Output must be a new directory: %s (%s)\n",
                ex->output_dir, strerror(errno));
        return -1;
    }
    snprintf(path, sizeof(path), "%s/motion_vectors", ex->output_dir);
    if (mkdir_if_needed(path) < 0)
        return -1;
    snprintf(path, sizeof(path), "%s/residuals", ex->output_dir);
    if (mkdir_if_needed(path) < 0)
        return -1;
    snprintf(path, sizeof(path), "%s/summary.csv", ex->output_dir);
    ex->summary = fopen(path, "w");
    if (!ex->summary) {
        fprintf(stderr, "Could not open %s: %s\n", path, strerror(errno));
        return -1;
    }
    fprintf(ex->summary,
            "frame,pict_type,mv_origin,mv_count,mean_abs_residual,"
            "width,height,native_mv_count,fallback_reason,pts,"
            "time_base_num,time_base_den\n");
    return 0;
}

static void motion_list_reset(MotionList *list)
{
    list->count = 0;
    list->origin = "none";
}

static int motion_list_append(MotionList *list, const MotionBlock *block)
{
    if (list->count == list->capacity) {
        int new_capacity = list->capacity ? list->capacity * 2 : 256;
        MotionBlock *new_items = (MotionBlock *)realloc(
            list->items, (size_t)new_capacity * sizeof(*new_items));
        if (!new_items)
            return AVERROR(ENOMEM);
        list->items = new_items;
        list->capacity = new_capacity;
    }
    list->items[list->count++] = *block;
    return 0;
}

static int clamp_int(int value, int low, int high)
{
    if (value < low)
        return low;
    if (value > high)
        return high;
    return value;
}

static int copy_native_motion_vectors(const AVFrame *frame, MotionList *list)
{
    AVFrameSideData *side_data = av_frame_get_side_data(
        frame, AV_FRAME_DATA_MOTION_VECTORS);
    int i;

    if (!side_data ||
        (size_t)side_data->size < sizeof(AVMotionVector))
        return 0;

    {
        const AVMotionVector *vectors =
            (const AVMotionVector *)side_data->data;
        int count = (int)(side_data->size / sizeof(*vectors));
        for (i = 0; i < count; ++i) {
            MotionBlock block;
            const AVMotionVector *mv = &vectors[i];
            block.source = mv->source;
            block.w = mv->w;
            block.h = mv->h;
            block.src_x = mv->src_x;
            block.src_y = mv->src_y;
            block.dst_x = mv->dst_x;
            block.dst_y = mv->dst_y;
            block.motion_x = mv->motion_x;
            block.motion_y = mv->motion_y;
            block.motion_scale = mv->motion_scale;
            block.flags = mv->flags;
            if (motion_list_append(list, &block) < 0)
                return AVERROR(ENOMEM);
        }
    }
    list->origin = "bitstream";
    return list->count;
}

static uint64_t block_sad(const uint8_t *current,
                          const uint8_t *previous,
                          int width,
                          int height,
                          int dst_x,
                          int dst_y,
                          int src_x,
                          int src_y,
                          int block_w,
                          int block_h)
{
    uint64_t sad = 0;
    int y;
    int x;
    (void)height;
    for (y = 0; y < block_h; y += 2) {
        const uint8_t *cur_row = current + (dst_y + y) * width + dst_x;
        const uint8_t *prev_row = previous + (src_y + y) * width + src_x;
        for (x = 0; x < block_w; x += 2) {
            int diff = (int)cur_row[x] - (int)prev_row[x];
            sad += (uint64_t)(diff < 0 ? -diff : diff);
        }
    }
    return sad;
}

static int estimate_motion_vectors(const uint8_t *current,
                                   const uint8_t *previous,
                                   int width,
                                   int height,
                                   int block_size,
                                   int search_radius,
                                   MotionList *list)
{
    int dst_y;
    int dst_x;
    for (dst_y = 0; dst_y < height; dst_y += block_size) {
        for (dst_x = 0; dst_x < width; dst_x += block_size) {
            int block_w = block_size;
            int block_h = block_size;
            int best_src_x = dst_x;
            int best_src_y = dst_y;
            uint64_t best_sad = UINT64_MAX;
            int dy;
            int dx;

            if (dst_x + block_w > width)
                block_w = width - dst_x;
            if (dst_y + block_h > height)
                block_h = height - dst_y;

            for (dy = -search_radius; dy <= search_radius; ++dy) {
                int src_y = dst_y + dy;
                if (src_y < 0 || src_y + block_h > height)
                    continue;
                for (dx = -search_radius; dx <= search_radius; ++dx) {
                    int src_x = dst_x + dx;
                    uint64_t sad;
                    if (src_x < 0 || src_x + block_w > width)
                        continue;
                    sad = block_sad(current, previous, width, height,
                                    dst_x, dst_y, src_x, src_y,
                                    block_w, block_h);
                    if (sad < best_sad) {
                        best_sad = sad;
                        best_src_x = src_x;
                        best_src_y = src_y;
                    }
                }
            }

            {
                MotionBlock block;
                block.source = -1;
                block.w = block_w;
                block.h = block_h;
                block.src_x = best_src_x + block_w / 2;
                block.src_y = best_src_y + block_h / 2;
                block.dst_x = dst_x + block_w / 2;
                block.dst_y = dst_y + block_h / 2;
                block.motion_x = best_src_x - dst_x;
                block.motion_y = best_src_y - dst_y;
                block.motion_scale = 1;
                block.flags = 0;
                if (motion_list_append(list, &block) < 0)
                    return AVERROR(ENOMEM);
            }
        }
    }
    list->origin = "estimated";
    return list->count;
}

static int has_future_reference(const MotionList *list)
{
    int i;
    for (i = 0; i < list->count; ++i) {
        if (list->items[i].source > 0)
            return 1;
    }
    return 0;
}

static void build_prediction(const uint8_t *previous,
                             uint8_t *prediction,
                             int width,
                             int height,
                             const MotionList *list)
{
    int i;
    memcpy(prediction, previous, (size_t)width * height);

    for (i = 0; i < list->count; ++i) {
        const MotionBlock *mv = &list->items[i];
        int dst_left = mv->dst_x - mv->w / 2;
        int dst_top = mv->dst_y - mv->h / 2;
        int src_left = mv->src_x - mv->w / 2;
        int src_top = mv->src_y - mv->h / 2;
        int y;
        int x;

        /*
         * Positive sources point to future reference frames. They cannot be
         * reconstructed from previous_gray in this streaming implementation.
         */
        if (mv->source > 0)
            continue;

        for (y = 0; y < mv->h; ++y) {
            int dst_y = dst_top + y;
            int src_y = src_top + y;
            if (dst_y < 0 || dst_y >= height ||
                src_y < 0 || src_y >= height)
                continue;
            for (x = 0; x < mv->w; ++x) {
                int dst_x = dst_left + x;
                int src_x = src_left + x;
                if (dst_x < 0 || dst_x >= width ||
                    src_x < 0 || src_x >= width)
                    continue;
                prediction[dst_y * width + dst_x] =
                    previous[src_y * width + src_x];
            }
        }
    }
}

static double compute_residual(const uint8_t *current,
                               const uint8_t *prediction,
                               uint8_t *visualization,
                               int16_t *raw,
                               int width,
                               int height)
{
    uint64_t absolute_sum = 0;
    size_t pixels = (size_t)width * height;
    size_t i;
    for (i = 0; i < pixels; ++i) {
        int residual = (int)current[i] - (int)prediction[i];
        int magnitude = residual < 0 ? -residual : residual;
        absolute_sum += (uint64_t)magnitude;
        raw[i] = (int16_t)residual;
        visualization[i] = (uint8_t)clamp_int(residual + 128, 0, 255);
    }
    return pixels ? (double)absolute_sum / (double)pixels : 0.0;
}

static int write_motion_csv(const Extractor *ex, const MotionList *list)
{
    char path[PATH_MAX + 80];
    FILE *file;
    int i;
    snprintf(path, sizeof(path), "%s/motion_vectors/frame_%06d.csv",
             ex->output_dir, ex->frame_number);
    file = fopen(path, "w");
    if (!file)
        return -1;
    fprintf(file,
            "origin,source,block_w,block_h,src_x,src_y,dst_x,dst_y,"
            "motion_x,motion_y,motion_scale,flags\n");
    for (i = 0; i < list->count; ++i) {
        const MotionBlock *mv = &list->items[i];
        fprintf(file,
                "%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,0x%" PRIx64 "\n",
                list->origin, mv->source, mv->w, mv->h,
                mv->src_x, mv->src_y, mv->dst_x, mv->dst_y,
                mv->motion_x, mv->motion_y, mv->motion_scale, mv->flags);
    }
    fclose(file);
    return 0;
}

static int write_residual_pgm(const Extractor *ex)
{
    char path[PATH_MAX + 80];
    FILE *file;
    size_t pixels = (size_t)ex->width * ex->height;
    snprintf(path, sizeof(path), "%s/residuals/frame_%06d.pgm",
             ex->output_dir, ex->frame_number);
    file = fopen(path, "wb");
    if (!file)
        return -1;
    fprintf(file, "P5\n%d %d\n255\n", ex->width, ex->height);
    if (fwrite(ex->residual_vis, 1, pixels, file) != pixels) {
        fclose(file);
        return -1;
    }
    fclose(file);

    snprintf(path, sizeof(path), "%s/residuals/frame_%06d.s16le",
             ex->output_dir, ex->frame_number);
    file = fopen(path, "wb");
    if (!file)
        return -1;
    {
        size_t i;
        for (i = 0; i < pixels; ++i) {
            uint16_t value = (uint16_t)ex->residual_raw[i];
            uint8_t bytes[2];
            bytes[0] = (uint8_t)(value & 0xff);
            bytes[1] = (uint8_t)(value >> 8);
            if (fwrite(bytes, 1, sizeof(bytes), file) != sizeof(bytes)) {
                fclose(file);
                return -1;
            }
        }
    }
    fclose(file);
    return 0;
}

static int convert_to_gray(Extractor *ex, const AVFrame *frame)
{
    uint8_t *dst_data[4] = {ex->current_gray, NULL, NULL, NULL};
    int dst_linesize[4] = {ex->width, 0, 0, 0};

    /* Rescaling only pixels would make the native vector coordinates invalid. */
    if (frame->width != ex->width || frame->height != ex->height) {
        fprintf(stderr, "Resolution changes within a video are not supported.\n");
        return AVERROR(EINVAL);
    }

    ex->sws = sws_getCachedContext(
        ex->sws,
        frame->width, frame->height, (enum AVPixelFormat)frame->format,
        ex->width, ex->height, AV_PIX_FMT_GRAY8,
        SWS_BILINEAR, NULL, NULL, NULL);
    if (!ex->sws)
        return AVERROR(EINVAL);

    sws_scale(ex->sws,
              (const uint8_t *const *)frame->data, frame->linesize,
              0, frame->height, dst_data, dst_linesize);
    return 0;
}

static int process_frame(Extractor *ex, MotionList *motion)
{
    double mean_absolute_residual = 0.0;
    char pict_type = av_get_picture_type_char(ex->frame->pict_type);
    int native_count;
    const char *fallback_reason = "none";

    ex->frame_number++;
    if (convert_to_gray(ex, ex->frame) < 0)
        return AVERROR(EINVAL);

    motion_list_reset(motion);
    native_count = copy_native_motion_vectors(ex->frame, motion);
    if (native_count < 0)
        return native_count;

    if (ex->has_previous) {
        /*
         * This streaming residual uses the previous displayed frame. A
         * bitstream B-frame vector can reference a future frame, which is not
         * available under that definition. Use one coherent estimated field
         * rather than silently mixing incompatible reference directions.
         */
        if (native_count == 0 || has_future_reference(motion)) {
            fallback_reason = native_count == 0 ? "no_side_data" : "future_reference";
            motion_list_reset(motion);
            int ret = estimate_motion_vectors(
                ex->current_gray, ex->previous_gray,
                ex->width, ex->height,
                ex->block_size, ex->search_radius, motion);
            if (ret < 0)
                return ret;
        }
        build_prediction(ex->previous_gray, ex->prediction,
                         ex->width, ex->height, motion);
        mean_absolute_residual = compute_residual(
            ex->current_gray, ex->prediction, ex->residual_vis,
            ex->residual_raw,
            ex->width, ex->height);
    } else {
        memset(ex->residual_vis, 128, (size_t)ex->width * ex->height);
        memset(ex->residual_raw, 0,
               (size_t)ex->width * ex->height * sizeof(*ex->residual_raw));
    }

    if (write_motion_csv(ex, motion) < 0 ||
        write_residual_pgm(ex) < 0)
        return AVERROR(errno ? errno : EIO);

    fprintf(ex->summary, "%d,%c,%s,%d,%.6f,%d,%d,%d,%s,%" PRId64 ",%d,%d\n",
            ex->frame_number, pict_type, motion->origin, motion->count,
            mean_absolute_residual, ex->width, ex->height, native_count,
            fallback_reason, ex->frame->best_effort_timestamp,
            ex->format->streams[ex->video_stream]->time_base.num,
            ex->format->streams[ex->video_stream]->time_base.den);
    if (fflush(ex->summary) != 0)
        return AVERROR(EIO);

    memcpy(ex->previous_gray, ex->current_gray,
           (size_t)ex->width * ex->height);
    ex->has_previous = 1;
    return 0;
}

static int decode_packet(Extractor *ex,
                         const AVPacket *packet,
                         MotionList *motion)
{
    int ret = avcodec_send_packet(ex->decoder, packet);
    if (ret < 0 && ret != AVERROR_EOF)
        return ret;

    while (ret >= 0) {
        ret = avcodec_receive_frame(ex->decoder, ex->frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            return 0;
        if (ret < 0)
            return ret;
        ret = process_frame(ex, motion);
        av_frame_unref(ex->frame);
        if (ret < 0)
            return ret;
        if (ex->max_frames > 0 && ex->frame_number >= ex->max_frames)
            return AVERROR_EOF;
    }
    return 0;
}

static int open_input(Extractor *ex, const char *input)
{
    int ret;
    AVDictionary *options = NULL;
#if LIBAVCODEC_VERSION_MAJOR >= 59
    const AVCodec *codec = NULL;
#else
    AVCodec *codec = NULL;
#endif

    ret = avformat_open_input(&ex->format, input, NULL, NULL);
    if (ret < 0)
        return ret;
    ret = avformat_find_stream_info(ex->format, NULL);
    if (ret < 0)
        return ret;
    ret = av_find_best_stream(ex->format, AVMEDIA_TYPE_VIDEO,
                              -1, -1, &codec, 0);
    if (ret < 0)
        return ret;
    ex->video_stream = ret;
    ex->decoder = avcodec_alloc_context3(codec);
    if (!ex->decoder)
        return AVERROR(ENOMEM);
    ret = avcodec_parameters_to_context(
        ex->decoder, ex->format->streams[ex->video_stream]->codecpar);
    if (ret < 0)
        return ret;

    av_dict_set(&options, "flags2", "+export_mvs", 0);
    ret = avcodec_open2(ex->decoder, codec, &options);
    av_dict_free(&options);
    if (ret < 0)
        return ret;

    ex->width = ex->decoder->width;
    ex->height = ex->decoder->height;
    return av_image_check_size(ex->width, ex->height, 0, NULL);
}

static void cleanup(Extractor *ex, MotionList *motion)
{
    if (ex->summary)
        fclose(ex->summary);
    sws_freeContext(ex->sws);
    av_packet_free(&ex->packet);
    av_frame_free(&ex->frame);
    avcodec_free_context(&ex->decoder);
    avformat_close_input(&ex->format);
    free(ex->current_gray);
    free(ex->previous_gray);
    free(ex->prediction);
    free(ex->residual_vis);
    free(ex->residual_raw);
    free(motion->items);
}

static void usage(const char *program)
{
    fprintf(stderr,
            "Usage: %s <input-video> <output-dir> "
            "[max-frames=0] [block-size=16] [search-radius=8]\n",
            program);
}

static int parse_integer(const char *text, int minimum, int maximum, int *value)
{
    char *end = NULL;
    long number;
    errno = 0;
    number = strtol(text, &end, 10);
    if (errno || end == text || *end || number < minimum || number > maximum)
        return -1;
    *value = (int)number;
    return 0;
}

int main(int argc, char **argv)
{
    Extractor ex;
    MotionList motion;
    int ret;
    size_t pixels;

    memset(&ex, 0, sizeof(ex));
    memset(&motion, 0, sizeof(motion));
    ex.video_stream = -1;
    ex.block_size = 16;
    ex.search_radius = 8;

    if (argc == 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h"))) {
        usage(argv[0]);
        return 0;
    }
    if (argc < 3 || argc > 6) {
        usage(argv[0]);
        return 2;
    }
    if (!argv[2][0] || strlen(argv[2]) >= sizeof(ex.output_dir) - 80) {
        fprintf(stderr, "Output path is empty or too long.\n");
        return 2;
    }
    snprintf(ex.output_dir, sizeof(ex.output_dir), "%s", argv[2]);
    if ((argc >= 4 && parse_integer(argv[3], 0, INT_MAX, &ex.max_frames) < 0) ||
        (argc >= 5 && parse_integer(argv[4], 1, 2048, &ex.block_size) < 0) ||
        (argc >= 6 && parse_integer(argv[5], 0, 256, &ex.search_radius) < 0)) {
        usage(argv[0]);
        return 2;
    }

    ret = open_input(&ex, argv[1]);
    if (ret < 0) {
        print_av_error("Could not open video", ret);
        cleanup(&ex, &motion);
        return 1;
    }
    if (prepare_output_directories(&ex) < 0) {
        cleanup(&ex, &motion);
        return 1;
    }

    pixels = (size_t)ex.width * ex.height;
    ex.current_gray = (uint8_t *)malloc(pixels);
    ex.previous_gray = (uint8_t *)malloc(pixels);
    ex.prediction = (uint8_t *)malloc(pixels);
    ex.residual_vis = (uint8_t *)malloc(pixels);
    ex.residual_raw = (int16_t *)malloc(pixels * sizeof(*ex.residual_raw));
    ex.frame = av_frame_alloc();
    ex.packet = av_packet_alloc();
    if (!ex.current_gray || !ex.previous_gray || !ex.prediction ||
        !ex.residual_vis || !ex.residual_raw || !ex.frame || !ex.packet) {
        fprintf(stderr, "Out of memory\n");
        cleanup(&ex, &motion);
        return 1;
    }

    while ((ret = av_read_frame(ex.format, ex.packet)) >= 0) {
        if (ex.packet->stream_index == ex.video_stream) {
            ret = decode_packet(&ex, ex.packet, &motion);
            if (ret == AVERROR_EOF) {
                av_packet_unref(ex.packet);
                ret = 0;
                break;
            }
            if (ret < 0) {
                print_av_error("Decode failed", ret);
                av_packet_unref(ex.packet);
                break;
            }
        }
        av_packet_unref(ex.packet);
    }
    if ((ex.max_frames <= 0 || ex.frame_number < ex.max_frames) &&
        (ret >= 0 || ret == AVERROR_EOF))
        ret = decode_packet(&ex, NULL, &motion);
    if (ret == AVERROR_EOF)
        ret = 0;

    if (ex.frame_number == 0 && ret == 0) {
        fprintf(stderr, "No video frames were decoded.\n");
        ret = AVERROR_INVALIDDATA;
    }

    fprintf(stderr,
            "Extracted %d frames from codec=%s into %s\n",
            ex.frame_number,
            ex.decoder && ex.decoder->codec ?
                ex.decoder->codec->name : "unknown",
            ex.output_dir);
    cleanup(&ex, &motion);
    return ret < 0;
}
