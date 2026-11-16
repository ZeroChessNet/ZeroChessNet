use crate::{Board, Move, SearchContext, ZcnEngine, ZcnEvaluator};
use std::io::{self, BufRead, Write};

pub struct UciTerminal<E: ZcnEngine> {
    pub engine: E,
    pub board: Board,
    pub eval: ZcnEvaluator,
    pub hash_mb: usize,
    pub threads: usize,
}

impl<E: ZcnEngine> UciTerminal<E> {
    pub fn new(engine: E) -> Self {
        Self {
            engine,
            board: Board::default(),
            eval: ZcnEvaluator::new(),
            hash_mb: 64,
            threads: 1,
        }
    }

    pub fn run(&mut self) {
        let stdin = io::stdin();
        let mut stdout = io::stdout();

        for line in stdin.lock().lines().map_while(Result::ok) {
            let trimmed = line.trim();
            if trimmed.is_empty() { continue; }

            let cmd = trimmed.split_whitespace().next().unwrap_or("");

            match cmd {
                "uci" => {
                    writeln!(stdout, "id name {}", self.engine.name()).ok();
                    writeln!(stdout, "id author {}", self.engine.author()).ok();
                    writeln!(stdout, "option name Hash type spin default 64 min 1 max 1024").ok();
                    writeln!(stdout, "option name Threads type spin default 1 min 1 max 64").ok();
                    writeln!(stdout, "option name EvalFile type string default <empty>").ok();
                    if let Some(p) = self.eval.loaded_file() {
                        writeln!(stdout, "info string NNUE active: {}", p).ok();
                    }
                    writeln!(stdout, "uciok").ok();
                }
                "isready" => {
                    writeln!(stdout, "readyok").ok();
                }
                "ucinewgame" => {
                    self.board = Board::default();
                    self.engine.new_game();
                }
                "setoption" => {
                    self.handle_setoption(trimmed, &mut stdout);
                }
                "position" => {
                    self.handle_position(trimmed);
                }
                "go" => {
                    let ctx = SearchContext::from_uci_go(trimmed, self.board.white_to_move);
                    let best = self.engine.search(&self.board, &self.eval, &ctx);
                    writeln!(stdout, "bestmove {}", best.to_uci()).ok();
                }
                "d" => {
                    writeln!(stdout, "{}", self.board.display()).ok();
                }
                "eval" => {
                    let cp = self.eval.evaluate(&self.board);
                    let mode = if self.eval.loaded_file().is_some() { "TZNUE001 NNUE" } else { "PeSTO Fallback" };
                    writeln!(stdout, "Eval ({}): {} cp (side to move)", mode, cp).ok();
                }
                "quit" => break,
                _ => {}
            }
            stdout.flush().ok();
        }
    }

    fn handle_setoption(&mut self, line: &str, stdout: &mut io::Stdout) {
        let lower = line.to_ascii_lowercase();
        if let Some(val_pos) = lower.find(" value ") {
            let name_part = lower[..val_pos].trim();
            let value_str = line[val_pos + 7..].trim();

            if name_part.ends_with("evalfile") {
                match self.eval.load_nnue(value_str) {
                    Ok(()) => { writeln!(stdout, "info string Loaded NNUE: {}", value_str).ok(); }
                    Err(e) => { writeln!(stdout, "info string NNUE load failed: {}", e).ok(); }
                }
            } else if name_part.ends_with("hash") {
                if let Ok(mb) = value_str.parse::<usize>() {
                    self.hash_mb = mb.clamp(1, 1024);
                }
            } else if name_part.ends_with("threads") {
                if let Ok(t) = value_str.parse::<usize>() {
                    self.threads = t.clamp(1, 64);
                }
            }
        }
    }

    fn handle_position(&mut self, line: &str) {
        let rest = line.strip_prefix("position").unwrap_or("").trim();
        let (pos_part, moves_part) = match rest.find("moves") {
            Some(idx) => (&rest[..idx], Some(&rest[idx + 5..])),
            None => (rest, None),
        };

        let pos_trimmed = pos_part.trim();
        if pos_trimmed.starts_with("startpos") {
            self.board = Board::default();
        } else if let Some(fen) = pos_trimmed.strip_prefix("fen") {
            if let Ok(b) = Board::from_fen(fen.trim()) {
                self.board = b;
            }
        }

        if let Some(mvs) = moves_part {
            for m_str in mvs.split_whitespace() {
                if let Some(mv) = Move::from_uci(m_str) {
                    self.board.apply_uci_move(mv);
                }
            }
        }
    }
}
