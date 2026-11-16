use crate::{Board, Move, SearchContext, ZcnEngine, ZcnEvaluator};
use std::io::{self, BufRead, Write};

pub struct UciTerminal<E: ZcnEngine> {
    pub engine: E,
    pub board: Board,
    pub eval: ZcnEvaluator,
    pub hash_mb: usize,
    pub threads: usize,
    engine_name: String,
    engine_author: String,
}

impl<E: ZcnEngine> UciTerminal<E> {
    pub fn new(engine: E) -> Self {
        let engine_name = engine.name().to_owned();
        let engine_author = engine.author().to_owned();
        Self {
            engine,
            board: Board::default(),
            eval: ZcnEvaluator::new(),
            hash_mb: 64,
            threads: 1,
            engine_name,
            engine_author,
        }
    }

    pub fn with_identity(
        mut self,
        name: impl Into<String>,
        author: impl Into<String>,
    ) -> Self {
        self.engine_name = name.into();
        self.engine_author = author.into();
        self
    }

    pub fn run(&mut self) -> io::Result<()> {
        let stdin = io::stdin();
        let mut stdout = io::stdout().lock();

        for line in stdin.lock().lines() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            match trimmed.split_whitespace().next().unwrap_or("") {
                "uci" => {
                    writeln!(stdout, "id name {}", self.engine_name)?;
                    writeln!(stdout, "id author {}", self.engine_author)?;
                    writeln!(stdout, "option name Hash type spin default 64 min 1 max 1024")?;
                    writeln!(stdout, "option name Threads type spin default 1 min 1 max 64")?;
                    writeln!(stdout, "option name EvalFile type string default <empty>")?;
                    if let Some(path) = self.eval.loaded_file() {
                        writeln!(stdout, "info string NNUE active: {path}")?;
                    }
                    writeln!(stdout, "uciok")?;
                }
                "isready" => writeln!(stdout, "readyok")?,
                "ucinewgame" => {
                    self.board = Board::default();
                    self.engine.new_game();
                }
                "setoption" => self.handle_setoption(trimmed, &mut stdout)?,
                "position" => self.handle_position(trimmed, &mut stdout)?,
                "go" => {
                    let context =
                        SearchContext::from_uci_go(trimmed, self.board.white_to_move());
                    let best = self.engine.search(&self.board, &self.eval, &context);
                    let best = if best.is_none() || self.board.legal_moves().contains(&best) {
                        best
                    } else {
                        writeln!(stdout, "info string engine returned an illegal move")?;
                        Move::NONE
                    };
                    writeln!(stdout, "bestmove {}", best.to_uci())?;
                }
                "d" => write!(stdout, "{}", self.board.display())?,
                "eval" => {
                    let centipawns = self.eval.evaluate(&self.board);
                    let mode = if self.eval.loaded_file().is_some() {
                        "TZNUE001 NNUE"
                    } else {
                        "material fallback"
                    };
                    writeln!(
                        stdout,
                        "Eval ({mode}): {centipawns} cp (side to move)"
                    )?;
                }
                "quit" => break,
                _ => {}
            }
            stdout.flush()?;
        }
        Ok(())
    }

    fn handle_setoption(&mut self, line: &str, stdout: &mut impl Write) -> io::Result<()> {
        let lower = line.to_ascii_lowercase();
        let Some(value_pos) = lower.find(" value ") else {
            writeln!(stdout, "info string invalid setoption command")?;
            return Ok(());
        };
        let name_part = lower[..value_pos].trim();
        let value = line[value_pos + " value ".len()..].trim();

        if name_part.starts_with("setoption name ") && name_part.ends_with("evalfile") {
            match self.eval.load_nnue(value) {
                Ok(()) => writeln!(stdout, "info string Loaded NNUE: {value}")?,
                Err(error) => writeln!(stdout, "info string NNUE load failed: {error}")?,
            }
        } else if name_part.starts_with("setoption name ") && name_part.ends_with("hash") {
            match value.parse::<usize>() {
                Ok(mb) => self.hash_mb = mb.clamp(1, 1024),
                Err(_) => writeln!(stdout, "info string invalid Hash value: {value}")?,
            }
        } else if name_part.starts_with("setoption name ")
            && name_part.ends_with("threads")
        {
            match value.parse::<usize>() {
                Ok(threads) => self.threads = threads.clamp(1, 64),
                Err(_) => writeln!(stdout, "info string invalid Threads value: {value}")?,
            }
        }
        Ok(())
    }

    fn handle_position(&mut self, line: &str, stdout: &mut impl Write) -> io::Result<()> {
        let Some(rest) = line.strip_prefix("position") else {
            writeln!(stdout, "info string invalid position command")?;
            return Ok(());
        };
        let rest = rest.trim();
        let (position, moves) = match rest.split_once(" moves ") {
            Some((position, moves)) => (position, Some(moves)),
            None => (rest, None),
        };

        let mut next = if position == "startpos" {
            Board::default()
        } else if let Some(fen) = position.strip_prefix("fen ") {
            match Board::from_fen(fen) {
                Ok(board) => board,
                Err(error) => {
                    writeln!(stdout, "info string invalid FEN: {error}")?;
                    return Ok(());
                }
            }
        } else {
            writeln!(stdout, "info string invalid position command")?;
            return Ok(());
        };

        if let Some(moves) = moves {
            for text in moves.split_whitespace() {
                let Some(mv) = Move::from_uci(text) else {
                    writeln!(stdout, "info string invalid move: {text}")?;
                    return Ok(());
                };
                if next.try_apply_move(mv).is_err() {
                    writeln!(stdout, "info string illegal move: {text}")?;
                    return Ok(());
                }
            }
        }
        self.board = next;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StubEngine;

    impl ZcnEngine for StubEngine {
        fn search(&mut self, _: &Board, _: &ZcnEvaluator, _: &SearchContext) -> Move {
            Move::NONE
        }
    }

    #[test]
    fn default_engine_identity_is_available_and_overridable() {
        let terminal = UciTerminal::new(StubEngine);
        assert_eq!(terminal.engine_name, "ZeroChessNet");
        assert_eq!(terminal.engine_author, "ZeroChessNet contributors");

        let terminal = terminal.with_identity("MyEngine", "MyAuthor");
        assert_eq!(terminal.engine_name, "MyEngine");
        assert_eq!(terminal.engine_author, "MyAuthor");
    }

    #[test]
    fn invalid_position_does_not_replace_current_board() {
        let mut terminal = UciTerminal::new(StubEngine);
        let original = terminal.board.clone();
        let mut output = Vec::new();
        terminal
            .handle_position("position startpos moves e2e5", &mut output)
            .unwrap();
        assert_eq!(terminal.board, original);
        assert!(String::from_utf8(output).unwrap().contains("illegal move"));
    }
}
