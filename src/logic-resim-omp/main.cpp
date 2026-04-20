// logic-resim-omp/main.cpp
// OpenMP target offloading port of logic-resim-cuda: event-driven logic circuit simulator.

#include <omp.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <random>

static constexpr char value0 = 0;
static constexpr char value1 = 1;
static constexpr char valueX = 2;
static constexpr char valueZ = 3;

static constexpr int TRUTH_SIZE = 1025;

static constexpr int MAX_INPUT_PORT = 6;
static constexpr int MAX_DTUPLE     = 12;
static constexpr int MAX_DTABLE     = MAX_DTUPLE * MAX_INPUT_PORT;

typedef unsigned long long tUnit;
typedef unsigned int       dUnit;

struct Event {
    tUnit t;
    char  v;
};

struct SimGate {
    int   funcSer;
    int   iPortBase[MAX_INPUT_PORT];
    int   iPortLen [MAX_INPUT_PORT];
    dUnit dTable   [MAX_DTABLE];
};

#pragma omp declare target
static void simulate_gate(
    const SimGate* gates,
    const Event*   all_input_events,
    Event*         output_events,
    const char*    eTableMgr,
    const int*     out_base,
    int*           out_size,
    const int*     out_max,
    int*           overflow,
    tUnit dumpOff,
    int   gid)
{
    const SimGate& gate = gates[gid];

    const char* eTable = &eTableMgr[gate.funcSer * TRUTH_SIZE];
    const int8_t inLen = static_cast<int8_t>(*(eTable + TRUTH_SIZE - 1));

    const int maxSize = out_max[gid];
    const int oBase   = out_base[gid];
    Event* oPort      = &output_events[oBase];

    overflow[gid] = 0;

    int   iPortIdx[MAX_INPUT_PORT];
    tUnit iTnext  [MAX_INPUT_PORT];

    for (int i = 0; i < inLen; i++) {
        iPortIdx[i] = 0;
        iTnext[i]   = (gate.iPortLen[i] > 1)
                          ? all_input_events[gate.iPortBase[i] + 1].t
                          : static_cast<tUnit>(-1);
    }

    int16_t iVprev = 0;
    tUnit   currTime = static_cast<tUnit>(-1);
    bool    isInit   = true;

    for (int i = 0; i < inLen; i++) {
        iVprev  = static_cast<int16_t>(iVprev << 2);
        iVprev |= all_input_events[gate.iPortBase[i]].v;
        if (all_input_events[gate.iPortBase[i]].t > 0)
            isInit = false;
        if (iTnext[i] < currTime)
            currTime = iTnext[i];
    }

    int    currSize;
    int8_t oVprev;

    if (isInit) {
        currSize  = 1;
        oPort[0]  = {0, valueX};
        oVprev    = valueX;
    } else {
        currSize = out_size[gid];
        oVprev   = oPort[currSize - 1].v;
    }

    if (currTime == static_cast<tUnit>(-1) || currTime > dumpOff) {
        if (oPort[currSize - 1].t != static_cast<tUnit>(-1)) {
            oPort[currSize] = {static_cast<tUnit>(-1), valueZ};
            out_size[gid]   = currSize + 1;
        }
        return;
    }

    int16_t iVcurr = 0;
    int8_t  oVcurr = 0;
    dUnit   delay  = static_cast<dUnit>(-1);

    while (currTime != static_cast<tUnit>(-1) && currTime <= dumpOff) {
        for (int i = 0; i < inLen; ++i) {
            if (iTnext[i] == currTime) {
                ++iPortIdx[i];
                const int next = iPortIdx[i] + 1;
                iTnext[i] = (next < gate.iPortLen[i])
                                ? all_input_events[gate.iPortBase[i] + next].t
                                : static_cast<tUnit>(-1);
            }
            iVcurr  = static_cast<int16_t>(iVcurr << 2);
            iVcurr |= all_input_events[gate.iPortBase[i] + iPortIdx[i]].v;
        }

        oVcurr = (eTable[iVcurr >> 2] >> ((iVcurr & 3) * 2)) & 3;

        if (oVcurr != oVprev) {
            int delayIdx;
            switch (oVprev << 2 | oVcurr) {
                case 1: delayIdx = 0; break;
                case 4: delayIdx = 1; break;
                case 2: delayIdx = 2; break;
                case 9: delayIdx = 3; break;
                case 6: delayIdx = 4; break;
                case 8: delayIdx = 5; break;
                default: delayIdx = 0; break;
            }

            delay = static_cast<dUnit>(-1);
            for (int i = 0; i < inLen; ++i) {
                int iTo   = (iVcurr >> ((inLen - i - 1) << 1)) & 3;
                int iFrom = (iVprev >> ((inLen - i - 1) << 1)) & 3;
                if (iTo != iFrom) {
                    dUnit d = (iFrom == 1 || iTo == 0)
                                  ? gate.dTable[MAX_DTUPLE * i + delayIdx + 6]
                                  : gate.dTable[MAX_DTUPLE * i + delayIdx];
                    if (d < delay)
                        delay = d;
                }
            }

            currTime += delay;

            while (currSize && oPort[currSize - 1].t >= currTime)
                --currSize;

            if (!currSize || oPort[currSize - 1].v != oVcurr) {
                oPort[currSize] = {currTime, static_cast<char>(oVcurr)};
                ++currSize;
                if (currSize == maxSize) {
                    overflow[gid]  = 1;
                    out_size[gid]  = currSize;
                    return;
                }
            }
        }

        iVprev  = iVcurr;
        oVprev  = oVcurr;
        iVcurr  = 0;
        oVcurr  = 0;
        currTime = static_cast<tUnit>(-1);
        delay    = static_cast<dUnit>(-1);

        for (int i = 0; i < inLen; ++i)
            if (iTnext[i] < currTime)
                currTime = iTnext[i];
    }

    oPort[currSize] = {static_cast<tUnit>(-1), valueZ};
    out_size[gid]   = currSize + 1;
}
#pragma omp end declare target

static void buildAndGateTruthTable(std::vector<char>& table) {
    table.assign(TRUTH_SIZE, 0);
    for (int iVcurr = 0; iVcurr < 16; ++iVcurr) {
        int in0 = (iVcurr >> 2) & 3;
        int in1 =  iVcurr       & 3;
        char out;
        if (in0 == value0 || in1 == value0) out = value0;
        else if (in0 == value1 && in1 == value1) out = value1;
        else out = valueX;
        int byte_pos = iVcurr >> 2;
        int bit_off  = (iVcurr & 3) * 2;
        table[byte_pos] |= static_cast<char>(out << bit_off);
    }
    table[TRUTH_SIZE - 1] = 2;
}

int main(int argc, char* argv[]) {
    int simSize = 100;
    if (argc > 1) simSize = atoi(argv[1]);

    const int   maxInputPorts    = 2;
    const int   maxEventsPerWire = 64;
    const int   maxOutputEvents  = 256;
    const tUnit dumpOff          = 500000ULL;

    std::vector<char> h_eTableMgr(TRUTH_SIZE);
    buildAndGateTruthTable(h_eTableMgr);

    const int totalInputEvents = simSize * maxInputPorts * maxEventsPerWire;
    const int totalOutputEvts  = simSize * maxOutputEvents;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int>  val_dist(0, 1);
    std::uniform_int_distribution<int>  del_dist(100, 5000);

    std::vector<SimGate> h_gates(simSize);
    std::vector<Event>   h_input_events(totalInputEvents);
    std::vector<int>     h_out_base(simSize);
    std::vector<int>     h_out_max(simSize, maxOutputEvents);
    std::vector<int>     h_out_size(simSize, 1);

    int event_idx = 0;
    for (int g = 0; g < simSize; ++g) {
        h_gates[g].funcSer = 0;
        for (int p = 0; p < maxInputPorts; ++p) {
            h_gates[g].iPortBase[p] = event_idx;
            h_gates[g].iPortLen[p]  = maxEventsPerWire;
            tUnit t = 0;
            h_input_events[event_idx] = {0, static_cast<char>(val_dist(rng))};
            for (int e = 1; e < maxEventsPerWire - 1; ++e) {
                t += static_cast<tUnit>(del_dist(rng));
                h_input_events[event_idx + e] = {t, static_cast<char>(val_dist(rng))};
            }
            h_input_events[event_idx + maxEventsPerWire - 1] = {
                static_cast<tUnit>(-1), valueZ};
            event_idx += maxEventsPerWire;
        }
        for (int p = maxInputPorts; p < MAX_INPUT_PORT; ++p) {
            h_gates[g].iPortBase[p] = 0;
            h_gates[g].iPortLen[p]  = 0;
        }
        for (int d = 0; d < MAX_DTABLE; ++d)
            h_gates[g].dTable[d] = static_cast<dUnit>(del_dist(rng));
        h_out_base[g] = g * maxOutputEvents;
    }

    SimGate* d_gates         = (SimGate*)malloc(simSize * sizeof(SimGate));
    Event*   d_input_events  = (Event*)malloc(totalInputEvents * sizeof(Event));
    Event*   d_output_events = (Event*)malloc(totalOutputEvts * sizeof(Event));
    char*    d_eTableMgr     = (char*)malloc(TRUTH_SIZE * sizeof(char));
    int*     d_out_base      = (int*)malloc(simSize * sizeof(int));
    int*     d_out_max       = (int*)malloc(simSize * sizeof(int));
    int*     d_out_size      = (int*)malloc(simSize * sizeof(int));
    int*     d_overflow      = (int*)malloc(simSize * sizeof(int));

    memcpy(d_gates,        h_gates.data(),        simSize * sizeof(SimGate));
    memcpy(d_input_events, h_input_events.data(),  totalInputEvents * sizeof(Event));
    memcpy(d_eTableMgr,    h_eTableMgr.data(),     TRUTH_SIZE * sizeof(char));
    memcpy(d_out_base,     h_out_base.data(),       simSize * sizeof(int));
    memcpy(d_out_max,      h_out_max.data(),        simSize * sizeof(int));
    memcpy(d_out_size,     h_out_size.data(),       simSize * sizeof(int));
    memset(d_overflow,     0,                       simSize * sizeof(int));
    memset(d_output_events, 0,                      totalOutputEvts * sizeof(Event));

    #pragma omp target enter data \
        map(to: d_gates[0:simSize], d_input_events[0:totalInputEvents], \
                d_eTableMgr[0:TRUTH_SIZE], d_out_base[0:simSize], \
                d_out_max[0:simSize], d_out_size[0:simSize], d_overflow[0:simSize]) \
        map(alloc: d_output_events[0:totalOutputEvts])

    printf("Logic re-simulation: %d 2-input AND gates, dumpOff=%llu\n",
           simSize, (unsigned long long)dumpOff);

    auto t0 = std::chrono::steady_clock::now();

    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int gid = 0; gid < simSize; gid++) {
        simulate_gate(d_gates, d_input_events, d_output_events,
                      d_eTableMgr, d_out_base, d_out_size, d_out_max,
                      d_overflow, dumpOff, gid);
    }

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    printf("Total simulation time: %.6f (s)\n", elapsed);

    #pragma omp target update from(d_overflow[0:simSize], d_out_size[0:simSize])

    int overflowed   = 0;
    long long total_events = 0;
    for (int g = 0; g < simSize; ++g) {
        overflowed   += d_overflow[g];
        total_events += d_out_size[g];
    }
    printf("Overflowed: %d/%d gates, total output events: %lld\n",
           overflowed, simSize, total_events);

    #pragma omp target exit data \
        map(delete: d_gates[0:simSize], d_input_events[0:totalInputEvents], \
                    d_output_events[0:totalOutputEvts], d_eTableMgr[0:TRUTH_SIZE], \
                    d_out_base[0:simSize], d_out_max[0:simSize], \
                    d_out_size[0:simSize], d_overflow[0:simSize])

    free(d_gates); free(d_input_events); free(d_output_events);
    free(d_eTableMgr); free(d_out_base); free(d_out_max);
    free(d_out_size); free(d_overflow);
    return 0;
}
