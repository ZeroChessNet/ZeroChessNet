#![forbid(unsafe_code)]

pub mod eval;
pub mod time;
pub mod uci;

pub use eval::ZcnEvaluator;
pub use time::SearchContext;
pub use uci::UciTerminal;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Piece {
    WhitePawn = 0, WhiteKnight = 1, WhiteBishop = 2, WhiteRook = 3, WhiteQueen = 4, WhiteKing = 5,
    BlackPawn = 6, BlackKnight = 7, BlackBishop = 8, BlackRook = 9, BlackQueen = 10, BlackKing = 11,
}

impl Piece {
    pub fn from_char(c: char) -> Option<Self> {
        match c {
            'P' => Some(Self::WhitePawn),   'N' => Some(Self::WhiteKnight),
            'B' => Some(Self::WhiteBishop), 'R' => Some(Self::WhiteRook),
            'Q' => Some(Self::WhiteQueen),  'K' => Some(Self::WhiteKing),
            'p' => Some(Self::BlackPawn),   'n' => Some(Self::BlackKnight),
            'b' => Some(Self::BlackBishop), 'r' => Some(Self::BlackRook),
            'q' => Some(Self::BlackQueen),  'k' => Some(Self::BlackKing),
            _ => None,
        }
    }

    pub fn to_char(self) -> char {
        const CHARS: [char; 12] = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k'];
        CHARS[self as usize]
    }

    pub fn is_white(self) -> bool {
        (self as u8) < 6
    }

    pub fn mirror_color(self) -> Self {
        const MIRROR: [Piece; 12] = [
            Piece::BlackPawn, Piece::BlackKnight, Piece::BlackBishop, Piece::BlackRook, Piece::BlackQueen, Piece::BlackKing,
            Piece::WhitePawn, Piece::WhiteKnight, Piece::WhiteBishop, Piece::WhiteRook, Piece::WhiteQueen, Piece::WhiteKing,
        ];
        MIRROR[self as usize]
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Move(pub u16);

impl Move {
    pub const NONE: Move = Move(0);

    pub fn new(from: u8, to: u8, promo: u8) -> Self {
        Move((from as u16) | ((to as u16) << 6) | ((promo as u16) << 12))
    }

    pub fn from_sq(self) -> u8 { (self.0 & 0x3F) as u8 }
    pub fn to_sq(self) -> u8 { ((self.0 >> 6) & 0x3F) as u8 }
    pub fn promo(self) -> u8 { ((self.0 >> 12) & 0x0F) as u8 }

    pub fn from_uci(uci: &str) -> Option<Self> {
        let b = uci.as_bytes();
        if b.len() < 4 { return None; }
        let f_file = b.wrapping_sub(b'a');
        let f_rank = b.wrapping_sub(b'1');
        let t_file = b.wrapping_sub(b'a');
        let t_rank = b.wrapping_sub(b'1');
        if f_file > 7 || f_rank > 7 || t_file > 7 || t_rank > 7 { return None; }
        let promo = match b.get(4).copied() {
            Some(b'q') | Some(b'Q') => 4,
            Some(b'r') | Some(b'R') => 3,
            Some(b'b') | Some(b'B') => 2,
            Some(b'n') | Some(b'N') => 1,
            _ => 0,
        };
        Some(Move::new(f_rank * 8 + f_file, t_rank * 8 + t_file, promo))
    }

    pub fn to_uci(self) -> String {
        if self == Move::NONE { return "0000".to_string(); }
        let f = self.from_sq();
        let t = self.to_sq();
        let mut s = String::with_capacity(5);
        s.push((b'a' + (f % 8)) as char);
        s.push((b'1' + (f / 8)) as char);
        s.push((b'a' + (t % 8)) as char);
        s.push((b'1' + (t / 8)) as char);
        match self.promo() {
            1 => s.push('n'),
            2 => s.push('b'),
            3 => s.push('r'),
            4 => s.push('q'),
            _ => {}
        }
        s
    }
}

#[derive(Clone, Debug)]
pub struct Board {
    pub squares: [Option<Piece>; 64],
    pub white_to_move: bool,
    pub castling: u8,
    pub ep_square: Option<u8>,
    pub halfmove_clock: u16,
    pub fullmove_number: u16,
}

impl Default for Board {
    fn default() -> Self {
        Self::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap()
    }
}

impl Board {
    pub fn from_fen(fen: &str) -> Result<Self, &'static str> {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() < 2 { return Err("Invalid FEN: missing fields"); }

        let mut squares = [None; 64];
        let mut rank: i8 = 7;
        let mut file: i8 = 0;

        for ch in parts.chars() {
            if ch == '/' {
                rank -= 1;
                file = 0;
            } else if let Some(skip) = ch.to_digit(10) {
                file += skip as i8;
            } else if let Some(piece) = Piece::from_char(ch) {
                if !(0..8).contains(&rank) || !(0..8).contains(&file) {
                    return Err("Invalid FEN: board out of bounds");
                }
                squares[(rank * 8 + file) as usize] = Some(piece);
                file += 1;
            } else {
                return Err("Invalid FEN: unknown piece character");
            }
        }

        let white_to_move = match parts {
            "w" => true,
            "b" => false,
            _ => return Err("Invalid FEN: side to move"),
        };

        let mut castling = 0u8;
        if let Some(&c_str) = parts.get(2) {
            for c in c_str.chars() {
                match c {
                    'K' => castling |= 1,
                    'Q' => castling |= 2,
                    'k' => castling |= 4,
                    'q' => castling |= 8,
                    _ => {}
                }
            }
        }

        let ep_square = parts.get(3).and_then(|&ep| {
            let b = ep.as_bytes();
            if b.len() == 2 && (b'a'..=b'h').contains(&b) && (b'1'..=b'8').contains(&b) {
                Some((b - b'1') * 8 + (b - b'a'))
            } else {
                None
            }
        });

        let halfmove_clock = parts.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
        let fullmove_number = parts.get(5).and_then(|s| s.parse().ok()).unwrap_or(1);

        Ok(Self { squares, white_to_move, castling, ep_square, halfmove_clock, fullmove_number })
    }

    pub fn apply_uci_move(&mut self, mv: Move) {
        let from = mv.from_sq() as usize;
        let to = mv.to_sq() as usize;
        let piece = match self.squares[from] {
            Some(p) => p,
            None => return,
        };

        self.squares[from] = None;

        if (piece == Piece::WhitePawn || piece == Piece::BlackPawn) && Some(to as u8) == self.ep_square {
            let cap_sq = if self.white_to_move { to - 8 } else { to + 8 };
            self.squares[cap_sq] = None;
        }

        if piece == Piece::WhiteKing && from == 4 {
            if to == 6 { self.squares = self.squares.take(); }
            if to == 2 { self.squares = self.squares.take(); }
        } else if piece == Piece::BlackKing && from == 60 {
            if to == 62 { self.squares = self.squares.take(); }
            if to == 58 { self.squares = self.squares.take(); }
        }

        let placed = match mv.promo() {
            1 => if self.white_to_move { Piece::WhiteKnight } else { Piece::BlackKnight },
            2 => if self.white_to_move { Piece::WhiteBishop } else { Piece::BlackBishop },
            3 => if self.white_to_move { Piece::WhiteRook }   else { Piece::BlackRook },
            4 => if self.white_to_move { Piece::WhiteQueen }  else { Piece::BlackQueen },
            _ => piece,
        };
        self.squares[to] = Some(placed);

        self.ep_square = if piece == Piece::WhitePawn && to == from + 16 {
            Some((from + 8) as u8)
        } else if piece == Piece::BlackPawn && from == to + 16 {
            Some((to + 8) as u8)
        } else {
            None
        };

        self.white_to_move = !self.white_to_move;
    }

    pub fn display(&self) -> String {
        let mut out = String::new();
        out.push_str("\n +---+---+---+---+---+---+---+---+\n");
        for rank in (0..8).rev() {
            out.push_str(" |");
            for file in 0..8 {
                let ch = self.squares[rank * 8 + file].map_or(' ', |p| p.to_char());
                out.push_str(&format!(" {} |", ch));
            }
            out.push_str(&format!(" {}\n +---+---+---+---+---+---+---+---+\n", rank + 1));
        }
        out.push_str("   a   b   c   d   e   f   g   h\n");
        out.push_str(&format!("Turn: {}\n", if self.white_to_move { "White" } else { "Black" }));
        out
    }
}

/// Plugin interface for any engine built on ZeroChessNet
pub trait ZcnEngine {
    fn name(&self) -> &str;
    fn author(&self) -> &str;
    fn new_game(&mut self) {}
    fn search(&mut self, board: &Board, eval: &ZcnEvaluator, ctx: &SearchContext) -> Move;
}
