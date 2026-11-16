# ZeroChessNet (`zcn-rust`)

**ZeroChessNet (ZCN)** is a zero-dependency, 100% safe-Rust (`#![forbid(unsafe_code)]`) chess engine framework that bundles a complete **UCI terminal**, **time management**, and a native **`TZNUE001` NNUE evaluation system** (with automatic PeSTO fallback) into a clean, reusable library.

Instead of rewriting UCI protocol parsers, FEN loaders, clock controllers, and neural network evaluators for every new bot, you simply implement a single trait (`ZcnEngine`) and plug your custom search algorithm straight into ZCN.

---

## Core Features

- **Zero External Dependencies & 100% Safe Rust:** Built exclusively with the Rust standard library and `#![forbid(unsafe_code)]`.
- **Complete UCI Terminal (`src/uci.rs`):** Handles `uci`, `isready`, `ucinewgame`, `setoption` (preserving spaces in Windows file paths), `position startpos/fen moves ...`, `go`, and `quit`.
- **Built-in Interactive Debug Commands:** Inspect positions directly in the terminal using `d` (ASCII board display) and `eval` (live NNUE or PeSTO centipawn evaluation).
- **Native `TZNUE001` NNUE Evaluator (`src/eval.rs`):** Loads dual-perspective `768 -> 256x2 -> 1` (`f32`) neural network weights via the UCI `EvalFile` option (or auto-loads `titan-zero.nnue` from the working directory), falling back cleanly to PeSTO material evaluation if no network is loaded.
- **Smart Clock & Deadline Management (`src/time.rs`):** Converts `wtime`, `btime`, `winc`, `binc`, `movetime`, and `depth` into soft and hard search deadlines with atomic stop signaling.
- **Modular 4-File Architecture:**
  - `src/lib.rs` — Core board state (`Board`), `Piece`, `Move`, and the `ZcnEngine` plugin trait.
  - `src/eval.rs` — `TZNUE001` binary weight loader, neural network inference, and PeSTO fallback.
  - `src/time.rs` — `SearchContext` time allocator and deadline checks.
  - `src/uci.rs` — `UciTerminal` event loop and command parser.

---

## Quick Start

### 1. Add `zero_chess_net` to Your Bot's `Cargo.toml`

Using a local path:
```toml
[dependencies]
zero_chess_net = { path = "../ZeroChessNet/zcn-rust" }
```

Or directly from GitHub:
```toml
[dependencies]
zero_chess_net = { git = "https://github.com" }
```

### 2. Implement `ZcnEngine` in `src/main.rs`

```rust
use zero_chess_net::{Board, Move, SearchContext, UciTerminal, ZcnEngine, ZcnEvaluator};

struct MyCustomBot;

impl ZcnEngine for MyCustomBot {
    fn name(&self) -> &str {
        "MyCustomBot 1.0"
    }

    fn author(&self) -> &str {
        "DadaK"
    }

    fn search(&mut self, board: &Board, eval: &ZcnEvaluator, ctx: &SearchContext) -> Move {
        // Evaluate the current board using TZNUE001 NNUE (or PeSTO fallback)
        let _static_score = eval.evaluate(board);

        // Check soft/hard time limits inside your iterative deepening loop
        if ctx.should_stop() {
            return Move::NONE;
        }

        // Replace with your search algorithm (PVS, Alpha-Beta, MCTS, etc.)
        Move::from_uci("e2e4").unwrap()
    }
}

fn main() {
    let mut terminal = UciTerminal::new(MyCustomBot);
    terminal.run();
}
```

---

## Interactive Terminal Commands

Run your compiled binary in any terminal or UCI GUI (such as **En Croissant**, **Cutechess**, or **Arena**):

| Command | Description |
| :--- | :--- |
| `uci` | Prints engine identity, UCI options (`Hash`, `Threads`, `EvalFile`), and active NNUE status. |
| `setoption name EvalFile value <path>` | Loads a `TZNUE001` `.nnue` weight file (supports spaces in file paths). |
| `position startpos moves e2e4 e7e5` | Sets up the starting position or a custom FEN and applies UCI moves. |
| `go wtime 60000 btime 60000` | Starts the engine search with automatic soft/hard deadline calculation. |
| `d` | Prints an ASCII diagram of the current board state and side to move. |
| `eval` | Displays the static evaluation in centipawns from the side-to-move perspective. |

---

## License

MIT License — free to use, modify, and integrate into your own chess engines.
