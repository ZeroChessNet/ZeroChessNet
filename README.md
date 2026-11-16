# ZeroChessNet (`zero_chess_net`)

ZeroChessNet is a safe-Rust chess framework for building UCI engines. It provides a UCI terminal, validated FEN handling, legal move generation/application, search-time management, and TZNUE001 NNUE evaluation with a material-evaluation fallback.

The default build has no third-party dependencies. Enable the optional `gpu` feature to run NNUE inference through `wgpu` on a compatible GPU.

## Features

- **Safe Rust:** the crate forbids unsafe code.
- **Legal chess moves:** generates legal moves, including castling, en passant, and all four promotion choices.
- **UCI terminal:** handles `uci`, `isready`, `ucinewgame`, `setoption`, `position`, `go`, `d`, `eval`, and `quit`.
- **NNUE evaluation:** loads the `TZNUE001` network format; uses material evaluation when no network is loaded.
- **Optional GPU inference:** `wgpu` compute support is available with the `gpu` feature. CPU inference remains the default.
- **Engine identity:** engines can override the default name and author through `ZcnEngine`, or set both on `UciTerminal`.
- **Search context:** calculates soft and hard limits from UCI clock and depth arguments, with an atomic stop flag.

## Installation

From crates.io:

```toml
[dependencies]
zero_chess_net = "1"
```

Or directly from this repository:

```toml
[dependencies]
zero_chess_net = { git = "https://github.com/ZeroChessNet/ZeroChessNet" }
```

Enable the optional GPU backend:

```toml
[dependencies]
zero_chess_net = { version = "1", features = ["gpu"] }
```

## Quick start

Implement `ZcnEngine`. The default identity is `ZeroChessNet` / `ZeroChessNet contributors`; override `name` and `author` as needed.

```rust
use std::io;
use zero_chess_net::{Board, Move, SearchContext, UciTerminal, ZcnEngine, ZcnEvaluator};

struct MyEngine;

impl ZcnEngine for MyEngine {
    fn name(&self) -> &str {
        "MyEngine 1.0"
    }

    fn author(&self) -> &str {
        "Example author"
    }

    fn search(
        &mut self,
        board: &Board,
        eval: &ZcnEvaluator,
        context: &SearchContext,
    ) -> Move {
        let _score = eval.evaluate(board);
        if context.should_stop() {
            return Move::NONE;
        }

        board
            .legal_moves()
            .into_iter()
            .next()
            .unwrap_or(Move::NONE)
    }
}

fn main() -> io::Result<()> {
    UciTerminal::new(MyEngine)
        .with_identity("MyEngine 1.0", "Example author")
        .run()
}
```

`Board::try_apply_move` rejects illegal moves. `Board::legal_moves` returns all legal moves for the current side to move.

## GPU NNUE inference

Build with `--features gpu`, load a network using `ZcnEvaluator::load_nnue`, then create a persistent `GpuEvaluator` for repeated inference:

```rust,ignore
use zero_chess_net::{Board, GpuEvaluator, ZcnEvaluator};

let mut evaluator = ZcnEvaluator::new();
evaluator.load_nnue("titan-zero.nnue")?;
let mut gpu = GpuEvaluator::new(&evaluator)?;
let score = gpu.evaluate(&Board::default())?;
println!("{}: {score} cp", gpu.adapter_name());
```

GPU support requires an available `wgpu`-compatible compute adapter. GPU initialization and inference return errors rather than silently switching backends. Call `ZcnEvaluator::evaluate` for CPU inference and automatic material fallback.

## NNUE format

The file consists of the eight-byte `TZNUE001` magic, a four-byte format header, then little-endian `f32` values in this order:

1. `768 × 256` feature weights
2. `256` feature biases
3. `512` output weights
4. `1` output bias

Files with missing, extra, or non-finite weight data are rejected. The loader only replaces the active network after the new file has been fully validated.

## Build and test

```sh
cargo test
cargo test --features gpu
cargo package --list
```

## License

MIT. See [LICENSE](LICENSE).
