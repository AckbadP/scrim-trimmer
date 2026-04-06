# To Implement
 - multiple smaller videos with following cases
    - clear case CD to WF
      - [x] video
      - [x] full image
      - [x] chat image
    - clear case CD to GF
      - [x] video
      - [x] full image
      - [x] chat image
    - case of CD to WF to CD to GF
      - [x] video
      - [x] full image
      - [x] chat image
    - case of WF to CD to WF
      - [x] video
      - [x] full image
      - [x] chat image
    - case of multiple variations of CD, WF, and GF (ie cd --cd-- \**cd\**, etc.)
      - [x] video
      - [x] full image
      - [x] chat image
    - base case of CD to WF surrounded by lots of random text
      - [x] video
      - [x] full image
      - [x] chat image
- single images encapsulating each of the above test cases
    - one just text box
    - one full screen
- [x] chat log files for each case

# Test Cases

Each test case has a video (`caseN.mkv`), chat log (`caseN.txt`), full screenshot
(`caseN_full.png`), and chat-only screenshot (`caseN_chat.png`, input box excluded).

The chat-only images are cropped from pixels (0, 312) to (192, 738) of the 1280×800
frame — the Local chat panel excluding the text input field at the bottom.

t0 is the EVE game time (seconds since midnight UTC) at video second 0.
Pass it to `parse_chat_logs(log_paths, t0, duration)`.

---

## Case 1 — Clear CD to WF

### ![case1_chat](case1_chat.png)
### ![case1_full](case1_full.png)

**Video**: `case1.mkv` (~41 s)  
**Chat log**: `case1.txt`  
**t0**: `17:43:55` UTC = **63835 s**

| Event | Game time | Video second |
|-------|-----------|-------------|
| CD    | 17:43:59  | 4 s         |
| wf    | 17:44:33  | 38 s        |

**Expected CDs**: `[4]`  
**Expected WFs**: `[38]`  
**Expected clips**: `[(4, 38)]`

---

## Case 2 — Clear CD to GF

### ![case2_chat](case2_chat.png)
### ![case2_full](case2_full.png)

**Video**: `case2.mkv` (~41 s)  
**Chat log**: `case2.txt`  
**t0**: `17:49:56` UTC = **64196 s**

| Event | Game time | Video second |
|-------|-----------|-------------|
| CD    | 17:50:01  | 5 s         |
| gf    | 17:50:31  | 35 s        |

`gf` counts as WF (`is_wf = first in ('WF', 'GF')`).

**Expected CDs**: `[5]`  
**Expected WFs**: `[35]`  
**Expected clips**: `[(5, 35)]`

---

## Case 3 — CD to WF to CD to GF

### ![case3_chat](case3_chat.png)
### ![case3_full](case3_full.png)

**Video**: `case3.mkv` (~100 s)  
**Chat log**: `case3.txt`  
**t0**: `17:52:22` UTC = **64342 s**

| Event | Game time | Video second |
|-------|-----------|-------------|
| CD    | 17:52:30  | 8 s         |
| wf    | 17:53:03  | 41 s        |
| CD    | 17:53:21  | 59 s        |
| gf    | 17:53:57  | 95 s        |

**Expected CDs**: `[8, 59]`  
**Expected WFs**: `[41, 95]`  
**Expected clips**: `[(8, 41), (59, 95)]`

---

## Case 4 — WF to CD to WF (initial WF before recording)

### ![case4_chat](case4_chat.png)
### ![case4_full](case4_full.png)

**Video**: `case4.mkv` (~38 s)  
**Chat log**: `case4.txt`  
**t0**: `17:55:48` UTC = **64548 s**

| Event | Game time | Video second | Note |
|-------|-----------|-------------|------|
| wf    | 17:55:32  | −16 s       | Before recording; visible in chat at t=0 but excluded (video_sec < 0) |
| CD    | 17:55:57  | 9 s         | |
| wf    | 17:56:21  | 33 s        | |

The initial `wf` at 17:55:32 was sent before recording started. It is visible in the
chat window at t=0 (the message persists on screen) but its video_sec is negative so
`parse_chat_logs` correctly excludes it.

**Expected CDs**: `[9]`  
**Expected WFs**: `[33]`  
**Expected clips**: `[(9, 33)]`

---

## Case 5 — Multiple CD / WF / GF Variations

### ![case5_chat](case5_chat.png)
### ![case5_full](case5_full.png)

**Video**: `case5.mkv` (~103 s)  
**Chat log**: `case5.txt`  
**t0**: `17:58:04` UTC = **64684 s**

### CD variants

| Message | Game time | Video second | Detected? |
|---------|-----------|-------------|-----------|
| `CD` | 17:58:10 | 6 s | ✓ |
| `cd` | 17:58:17 | 13 s | ✓ |
| `--CD--` | 17:58:26 | 22 s | ✓ — first word stripped of punctuation = `CD` |
| `--cd--` | 17:58:32 | 28 s | ✓ |
| `---------CD----------` | 17:58:39 | 35 s | ✓ |
| `------------cd------------` | 17:58:47 | 43 s | ✓ |
| `***** CD ******` | 17:58:56 | 52 s | ✓ |
| `***cd***` | 17:59:01 | 57 s | ✓ — single token, stripped = `cd` |

### WF / GF variants

| Message | Game time | Video second | Detected? |
|---------|-----------|-------------|-----------|
| `WF` | 17:59:09 | 65 s | ✓ |
| `wf` | 17:59:14 | 70 s | ✓ |
| `wf!` | 17:59:20 | 76 s | ✓ — `wf!` stripped = `wf` |
| `wf wf` | 17:59:28 | 84 s | ✓ — first word `wf` |
| `GF` | 17:59:34 | 90 s | ✓ |
| `gf` | 17:59:37 | 93 s | ✓ |
| `gf!` | 17:59:43 | 99 s | ✓ — `gf!` stripped = `gf` |

**Expected CDs**: `[6, 13, 22, 28, 35, 43, 52, 57]` (8 events)  
**Expected WFs**: `[65, 70, 76, 84, 90, 93, 99]` (7 events)  
**Expected clips**: `[(57, 65)]`  
After the first WF (65 s) all remaining WFs have no eligible CD → no further clips.

---

## Case 6 — CD to WF Surrounded by Random Text

### ![case6_chat](case6_chat.png)
### ![case6_full](case6_full.png)

**Video**: `case6.mkv` (~124 s)  
**Chat log**: `case6.txt`  
**t0**: `18:01:54` UTC = **64914 s**

Key events (random-text messages are omitted):

| Event | Game time | Video second | Note |
|-------|-----------|-------------|------|
| CD | 18:02:16 | 22 s | |
| CD | 18:03:07 | 73 s | |
| `…wf…` (embedded) | 18:03:38 | 104 s | ✗ — `wf` is not the first word |
| `…WF…` (embedded) | 18:03:39 | 105 s | ✗ — `WF` is not the first word |
| `wf` | 18:03:43 | 109 s | ✓ |
| `wf` | 18:03:47 | 113 s | ✓ |

The two messages at 18:03:38–39 contain `wf`/`WF` embedded mid-sentence; they are
not detected because `wf`/`WF` is not the first word after the `>` separator.

**Expected CDs**: `[22, 73]`  
**Expected WFs**: `[109, 113]`  
**Expected clips**: `[(73, 109)]`  
CD at 22 s is abandoned because CD at 73 s is more recent before wf at 109 s.
wf at 113 s has no eligible CD after wf at 109 s → no second clip.
