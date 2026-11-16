#![forbid(unsafe_code)]

pub mod eval;
pub mod time;
pub mod uci;

#[cfg(feature = "gpu")]
pub mod gpu;

pub use eval::ZcnEvaluator;
pub use time::SearchContext;
pub use uci::UciTerminal;

#[cfg(feature = "gpu")]
pub use gpu::{GpuError, GpuEvaluator};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Piece {
    WhitePawn = 0,
    WhiteKnight = 1,
    WhiteBishop = 2,
    WhiteRook = 3,
    WhiteQueen = 4,
    WhiteKing = 5,
    BlackPawn = 6,
    BlackKnight = 7,
    BlackBishop = 8,
    BlackRook = 9,
    BlackQueen = 10,
    BlackKing = 11,
}

impl Piece {
    pub fn from_char(c: char) -> Option<Self> {
        Some(match c {
            'P' => Self::WhitePawn,
            'N' => Self::WhiteKnight,
            'B' => Self::WhiteBishop,
            'R' => Self::WhiteRook,
            'Q' => Self::WhiteQueen,
            'K' => Self::WhiteKing,
            'p' => Self::BlackPawn,
            'n' => Self::BlackKnight,
            'b' => Self::BlackBishop,
            'r' => Self::BlackRook,
            'q' => Self::BlackQueen,
            'k' => Self::BlackKing,
            _ => return None,
        })
    }

    pub const fn to_char(self) -> char {
        const CHARS: [char; 12] = [
            'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k',
        ];
        CHARS[self as usize]
    }

    pub const fn is_white(self) -> bool {
        (self as u8) < 6
    }

    pub const fn mirror_color(self) -> Self {
        const MIRROR: [Piece; 12] = [
            Piece::BlackPawn,
            Piece::BlackKnight,
            Piece::BlackBishop,
            Piece::BlackRook,
            Piece::BlackQueen,
            Piece::BlackKing,
            Piece::WhitePawn,
            Piece::WhiteKnight,
            Piece::WhiteBishop,
            Piece::WhiteRook,
            Piece::WhiteQueen,
            Piece::WhiteKing,
        ];
        MIRROR[self as usize]
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Move(u16);

impl Move {
    pub const NONE: Move = Move(u16::MAX);

    pub fn new(from: u8, to: u8, promo: u8) -> Self {
        assert!(from < 64 && to < 64 && promo <= 4, "invalid move encoding");
        Move((from as u16) | ((to as u16) << 6) | ((promo as u16) << 12))
    }

    pub fn from_sq(self) -> u8 {
        (self.0 & 0x3F) as u8
    }

    pub fn to_sq(self) -> u8 {
        ((self.0 >> 6) & 0x3F) as u8
    }

    pub fn promo(self) -> u8 {
        ((self.0 >> 12) & 0x0F) as u8
    }

    pub const fn is_none(self) -> bool {
        self.0 == Self::NONE.0
    }

    pub fn from_uci(uci: &str) -> Option<Self> {
        let bytes = uci.as_bytes();
        if bytes.len() != 4 && bytes.len() != 5 {
            return None;
        }
        let square = |file: u8, rank: u8| -> Option<u8> {
            if (b'a'..=b'h').contains(&file) && (b'1'..=b'8').contains(&rank) {
                Some((rank - b'1') * 8 + (file - b'a'))
            } else {
                None
            }
        };
        let from = square(bytes[0], bytes[1])?;
        let to = square(bytes[2], bytes[3])?;
        let promo = if bytes.len() == 5 {
            match bytes[4].to_ascii_lowercase() {
                b'n' => 1,
                b'b' => 2,
                b'r' => 3,
                b'q' => 4,
                _ => return None,
            }
        } else {
            0
        };
        Some(Self::new(from, to, promo))
    }

    pub fn to_uci(self) -> String {
        if self.is_none() {
            return "0000".to_owned();
        }
        let from = self.from_sq();
        let to = self.to_sq();
        let mut result = String::with_capacity(5);
        result.push((b'a' + from % 8) as char);
        result.push((b'1' + from / 8) as char);
        result.push((b'a' + to % 8) as char);
        result.push((b'1' + to / 8) as char);
        match self.promo() {
            1 => result.push('n'),
            2 => result.push('b'),
            3 => result.push('r'),
            4 => result.push('q'),
            _ => {}
        }
        result
    }
}

const WHITE_KING_SIDE: u8 = 1;
const WHITE_QUEEN_SIDE: u8 = 2;
const BLACK_KING_SIDE: u8 = 4;
const BLACK_QUEEN_SIDE: u8 = 8;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Board {
    squares: [Option<Piece>; 64],
    white_to_move: bool,
    castling: u8,
    ep_square: Option<u8>,
    halfmove_clock: u16,
    fullmove_number: u16,
}

impl Default for Board {
    fn default() -> Self {
        Self::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .expect("built-in starting position FEN is valid")
    }
}

impl Board {
    pub fn from_fen(fen: &str) -> Result<Self, &'static str> {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() != 6 {
            return Err("Invalid FEN: expected six fields");
        }

        let mut squares = [None; 64];
        let ranks: Vec<&str> = parts[0].split('/').collect();
        if ranks.len() != 8 {
            return Err("Invalid FEN: expected eight ranks");
        }
        for (fen_rank, rank_text) in ranks.iter().enumerate() {
            let mut file = 0usize;
            for ch in rank_text.chars() {
                if let Some(empty) = ch.to_digit(10) {
                    if !(1..=8).contains(&empty) {
                        return Err("Invalid FEN: invalid empty-square count");
                    }
                    file += empty as usize;
                } else if let Some(piece) = Piece::from_char(ch) {
                    if file >= 8 {
                        return Err("Invalid FEN: rank is too wide");
                    }
                    let square = (7 - fen_rank) * 8 + file;
                    squares[square] = Some(piece);
                    file += 1;
                } else {
                    return Err("Invalid FEN: unknown piece character");
                }
                if file > 8 {
                    return Err("Invalid FEN: rank is too wide");
                }
            }
            if file != 8 {
                return Err("Invalid FEN: rank is not eight squares wide");
            }
        }

        let white_to_move = match parts[1] {
            "w" => true,
            "b" => false,
            _ => return Err("Invalid FEN: side to move"),
        };

        let mut castling = 0u8;
        if parts[2] != "-" {
            for right in parts[2].chars() {
                let bit = match right {
                    'K' => WHITE_KING_SIDE,
                    'Q' => WHITE_QUEEN_SIDE,
                    'k' => BLACK_KING_SIDE,
                    'q' => BLACK_QUEEN_SIDE,
                    _ => return Err("Invalid FEN: castling rights"),
                };
                if castling & bit != 0 {
                    return Err("Invalid FEN: duplicate castling right");
                }
                castling |= bit;
            }
        }

        let ep_square = if parts[3] == "-" {
            None
        } else {
            let bytes = parts[3].as_bytes();
            if bytes.len() != 2
                || !(b'a'..=b'h').contains(&bytes[0])
                || !(b'1'..=b'8').contains(&bytes[1])
                || (white_to_move && bytes[1] != b'6')
                || (!white_to_move && bytes[1] != b'3')
            {
                return Err("Invalid FEN: en passant square");
            }
            Some((bytes[1] - b'1') * 8 + (bytes[0] - b'a'))
        };

        let halfmove_clock = parts[4]
            .parse::<u16>()
            .map_err(|_| "Invalid FEN: halfmove clock")?;
        let fullmove_number = parts[5]
            .parse::<u16>()
            .ok()
            .filter(|number| *number > 0)
            .ok_or("Invalid FEN: fullmove number")?;

        Ok(Self {
            squares,
            white_to_move,
            castling,
            ep_square,
            halfmove_clock,
            fullmove_number,
        })
    }

    pub const fn white_to_move(&self) -> bool {
        self.white_to_move
    }

    pub const fn piece_at(&self, square: u8) -> Option<Piece> {
        if square < 64 {
            self.squares[square as usize]
        } else {
            None
        }
    }

    pub const fn halfmove_clock(&self) -> u16 {
        self.halfmove_clock
    }

    pub const fn fullmove_number(&self) -> u16 {
        self.fullmove_number
    }

    pub const fn castling_rights(&self) -> u8 {
        self.castling
    }

    pub const fn en_passant_square(&self) -> Option<u8> {
        self.ep_square
    }

    pub const fn squares(&self) -> &[Option<Piece>; 64] {
        &self.squares
    }

    pub fn to_fen(&self) -> String {
        let mut fen = String::new();
        for rank in (0..8).rev() {
            let mut empty = 0;
            for file in 0..8 {
                match self.squares[rank * 8 + file] {
                    Some(piece) => {
                        if empty > 0 {
                            fen.push(char::from(b'0' + empty));
                            empty = 0;
                        }
                        fen.push(piece.to_char());
                    }
                    None => empty += 1,
                }
            }
            if empty > 0 {
                fen.push(char::from(b'0' + empty));
            }
            if rank > 0 {
                fen.push('/');
            }
        }
        fen.push_str(if self.white_to_move { " w " } else { " b " });
        if self.castling == 0 {
            fen.push('-');
        } else {
            for (right, bit) in [
                ('K', WHITE_KING_SIDE),
                ('Q', WHITE_QUEEN_SIDE),
                ('k', BLACK_KING_SIDE),
                ('q', BLACK_QUEEN_SIDE),
            ] {
                if self.castling & bit != 0 {
                    fen.push(right);
                }
            }
        }
        fen.push(' ');
        if let Some(square) = self.ep_square {
            fen.push((b'a' + square % 8) as char);
            fen.push((b'1' + square / 8) as char);
        } else {
            fen.push('-');
        }
        fen.push_str(&format!(
            " {} {}",
            self.halfmove_clock, self.fullmove_number
        ));
        fen
    }

    pub fn legal_moves(&self) -> Vec<Move> {
        let side = self.white_to_move;
        let mut candidates = Vec::with_capacity(64);
        for from in 0..64 {
            let Some(piece) = self.squares[from] else {
                continue;
            };
            if piece.is_white() != side {
                continue;
            }
            match piece {
                Piece::WhitePawn | Piece::BlackPawn => {
                    self.generate_pawn_moves(from, piece.is_white(), &mut candidates)
                }
                Piece::WhiteKnight | Piece::BlackKnight => {
                    self.generate_step_moves(from, &KNIGHT_STEPS, &mut candidates)
                }
                Piece::WhiteKing | Piece::BlackKing => {
                    self.generate_step_moves(from, &KING_STEPS, &mut candidates);
                    self.generate_castling_moves(from, piece.is_white(), &mut candidates);
                }
                Piece::WhiteBishop | Piece::BlackBishop => {
                    self.generate_sliding_moves(from, &BISHOP_DIRECTIONS, &mut candidates)
                }
                Piece::WhiteRook | Piece::BlackRook => {
                    self.generate_sliding_moves(from, &ROOK_DIRECTIONS, &mut candidates)
                }
                Piece::WhiteQueen | Piece::BlackQueen => {
                    self.generate_sliding_moves(from, &QUEEN_DIRECTIONS, &mut candidates)
                }
            }
        }

        candidates
            .into_iter()
            .filter(|mv| {
                if matches!(
                    self.squares[mv.to_sq() as usize],
                    Some(Piece::WhiteKing | Piece::BlackKing)
                ) {
                    return false;
                }
                let mut next = self.clone();
                next.apply_unchecked(*mv);
                !next.is_in_check(side)
            })
            .collect()
    }

    pub fn is_in_check(&self, white: bool) -> bool {
        let king = if white {
            Piece::WhiteKing
        } else {
            Piece::BlackKing
        };
        let Some(square) = self
            .squares
            .iter()
            .position(|piece| *piece == Some(king))
        else {
            return true;
        };
        self.is_square_attacked(square, !white)
    }

    pub fn try_apply_move(&mut self, mv: Move) -> Result<(), &'static str> {
        if !self.legal_moves().contains(&mv) {
            return Err("Move is not legal in this position");
        }
        self.apply_unchecked(mv);
        Ok(())
    }

    pub fn apply_uci_move(&mut self, mv: Move) -> Result<(), &'static str> {
        self.try_apply_move(mv)
    }

    pub fn display(&self) -> String {
        let mut out = String::new();
        out.push_str("\n +---+---+---+---+---+---+---+---+\n");
        for rank in (0..8).rev() {
            out.push_str(" |");
            for file in 0..8 {
                let ch = self.squares[rank * 8 + file].map_or(' ', Piece::to_char);
                out.push_str(&format!(" {} |", ch));
            }
            out.push_str(&format!(
                " {}\n +---+---+---+---+---+---+---+---+\n",
                rank + 1
            ));
        }
        out.push_str("   a   b   c   d   e   f   g   h\n");
        out.push_str(&format!(
            "Turn: {}\n",
            if self.white_to_move { "White" } else { "Black" }
        ));
        out
    }

    fn generate_pawn_moves(&self, from: usize, white: bool, moves: &mut Vec<Move>) {
        let file = (from % 8) as i8;
        let rank = (from / 8) as i8;
        let direction = if white { 1 } else { -1 };
        let next_rank = rank + direction;
        if (0..8).contains(&next_rank) {
            let one = (next_rank * 8 + file) as usize;
            if self.squares[one].is_none() {
                self.push_pawn_move(from, one, next_rank, moves);
                let start_rank = if white { 1 } else { 6 };
                let two_rank = rank + direction * 2;
                let two = (two_rank * 8 + file) as usize;
                if rank == start_rank && self.squares[two].is_none() {
                    moves.push(Move::new(from as u8, two as u8, 0));
                }
            }
            for target_file in [file - 1, file + 1] {
                if !(0..8).contains(&target_file) {
                    continue;
                }
                let to = (next_rank * 8 + target_file) as usize;
                let target = self.squares[to];
                let is_ep_capture = self.ep_square == Some(to as u8)
                    && self.squares[if white { to - 8 } else { to + 8 }]
                        == Some(if white {
                            Piece::BlackPawn
                        } else {
                            Piece::WhitePawn
                        });
                if target.is_some_and(|piece| piece.is_white() != white) || is_ep_capture {
                    self.push_pawn_move(from, to, next_rank, moves);
                }
            }
        }
    }

    fn push_pawn_move(&self, from: usize, to: usize, rank: i8, moves: &mut Vec<Move>) {
        if rank == 0 || rank == 7 {
            for promotion in [4, 3, 2, 1] {
                moves.push(Move::new(from as u8, to as u8, promotion));
            }
        } else {
            moves.push(Move::new(from as u8, to as u8, 0));
        }
    }

    fn generate_step_moves(&self, from: usize, steps: &[(i8, i8)], moves: &mut Vec<Move>) {
        for &(df, dr) in steps {
            let file = (from % 8) as i8 + df;
            let rank = (from / 8) as i8 + dr;
            if let Some(to) = square_index(file, rank) {
                if self.squares[to].map_or(true, |piece| piece.is_white() != self.white_to_move) {
                    moves.push(Move::new(from as u8, to as u8, 0));
                }
            }
        }
    }

    fn generate_sliding_moves(
        &self,
        from: usize,
        directions: &[(i8, i8)],
        moves: &mut Vec<Move>,
    ) {
        for &(df, dr) in directions {
            let (mut file, mut rank) = ((from % 8) as i8 + df, (from / 8) as i8 + dr);
            while let Some(to) = square_index(file, rank) {
                match self.squares[to] {
                    None => moves.push(Move::new(from as u8, to as u8, 0)),
                    Some(piece) if piece.is_white() != self.white_to_move => {
                        moves.push(Move::new(from as u8, to as u8, 0));
                        break;
                    }
                    Some(_) => break,
                }
                file += df;
                rank += dr;
            }
        }
    }

    fn generate_castling_moves(&self, from: usize, white: bool, moves: &mut Vec<Move>) {
        let (king_from, king_side, queen_side, rook, rights) = if white {
            (4, 6, 2, Piece::WhiteRook, (WHITE_KING_SIDE, WHITE_QUEEN_SIDE))
        } else {
            (60, 62, 58, Piece::BlackRook, (BLACK_KING_SIDE, BLACK_QUEEN_SIDE))
        };
        if from != king_from || self.is_square_attacked(from, !white) {
            return;
        }

        let (king_right, queen_right) = rights;
        if self.castling & king_right != 0
            && self.squares[king_from + 1].is_none()
            && self.squares[king_from + 2].is_none()
            && self.squares[king_from + 3] == Some(rook)
            && !self.is_square_attacked(king_from + 1, !white)
            && !self.is_square_attacked(king_from + 2, !white)
        {
            moves.push(Move::new(from as u8, (king_from + 2) as u8, 0));
        }
        if self.castling & queen_right != 0
            && self.squares[king_from - 1].is_none()
            && self.squares[king_from - 2].is_none()
            && self.squares[king_from - 3].is_none()
            && self.squares[king_from - 4] == Some(rook)
            && !self.is_square_attacked(king_from - 1, !white)
            && !self.is_square_attacked(king_from - 2, !white)
        {
            moves.push(Move::new(from as u8, (king_from - 2) as u8, 0));
        }
    }

    fn is_square_attacked(&self, square: usize, by_white: bool) -> bool {
        let file = (square % 8) as i8;
        let rank = (square / 8) as i8;

        let pawn_rank = rank + if by_white { -1 } else { 1 };
        for pawn_file in [file - 1, file + 1] {
            if let Some(source) = square_index(pawn_file, pawn_rank) {
                if self.squares[source]
                    == Some(if by_white {
                        Piece::WhitePawn
                    } else {
                        Piece::BlackPawn
                    })
                {
                    return true;
                }
            }
        }

        for &(df, dr) in &KNIGHT_STEPS {
            if let Some(source) = square_index(file + df, rank + dr) {
                if self.squares[source]
                    == Some(if by_white {
                        Piece::WhiteKnight
                    } else {
                        Piece::BlackKnight
                    })
                {
                    return true;
                }
            }
        }

        for &(df, dr) in &KING_STEPS {
            if let Some(source) = square_index(file + df, rank + dr) {
                if self.squares[source]
                    == Some(if by_white {
                        Piece::WhiteKing
                    } else {
                        Piece::BlackKing
                    })
                {
                    return true;
                }
            }
        }

        for &(df, dr) in &QUEEN_DIRECTIONS {
            let diagonal = df != 0 && dr != 0;
            let (mut next_file, mut next_rank) = (file + df, rank + dr);
            while let Some(source) = square_index(next_file, next_rank) {
                if let Some(piece) = self.squares[source] {
                    if piece.is_white() == by_white {
                        let sliding_attack = match piece {
                            Piece::WhiteQueen | Piece::BlackQueen => true,
                            Piece::WhiteBishop | Piece::BlackBishop => diagonal,
                            Piece::WhiteRook | Piece::BlackRook => !diagonal,
                            _ => false,
                        };
                        if sliding_attack {
                            return true;
                        }
                    }
                    break;
                }
                next_file += df;
                next_rank += dr;
            }
        }
        false
    }

    fn apply_unchecked(&mut self, mv: Move) {
        let from = mv.from_sq() as usize;
        let to = mv.to_sq() as usize;
        let piece = self.squares[from].expect("generated move has a source piece");
        let captured = self.squares[to].is_some()
            || (matches!(piece, Piece::WhitePawn | Piece::BlackPawn)
                && self.ep_square == Some(to as u8)
                && from % 8 != to % 8);

        self.squares[from] = None;
        if matches!(piece, Piece::WhitePawn | Piece::BlackPawn)
            && self.ep_square == Some(to as u8)
            && from % 8 != to % 8
        {
            let captured_square = if piece.is_white() { to - 8 } else { to + 8 };
            self.squares[captured_square] = None;
        }

        match (piece, from, to) {
            (Piece::WhiteKing, 4, 6) => {
                self.squares[7] = None;
                self.squares[5] = Some(Piece::WhiteRook);
            }
            (Piece::WhiteKing, 4, 2) => {
                self.squares[0] = None;
                self.squares[3] = Some(Piece::WhiteRook);
            }
            (Piece::BlackKing, 60, 62) => {
                self.squares[63] = None;
                self.squares[61] = Some(Piece::BlackRook);
            }
            (Piece::BlackKing, 60, 58) => {
                self.squares[56] = None;
                self.squares[59] = Some(Piece::BlackRook);
            }
            _ => {}
        }

        let placed = match mv.promo() {
            1 if piece.is_white() => Piece::WhiteKnight,
            2 if piece.is_white() => Piece::WhiteBishop,
            3 if piece.is_white() => Piece::WhiteRook,
            4 if piece.is_white() => Piece::WhiteQueen,
            1 => Piece::BlackKnight,
            2 => Piece::BlackBishop,
            3 => Piece::BlackRook,
            4 => Piece::BlackQueen,
            _ => piece,
        };
        self.squares[to] = Some(placed);

        match piece {
            Piece::WhiteKing => self.castling &= !(WHITE_KING_SIDE | WHITE_QUEEN_SIDE),
            Piece::BlackKing => self.castling &= !(BLACK_KING_SIDE | BLACK_QUEEN_SIDE),
            Piece::WhiteRook if from == 0 => self.castling &= !WHITE_QUEEN_SIDE,
            Piece::WhiteRook if from == 7 => self.castling &= !WHITE_KING_SIDE,
            Piece::BlackRook if from == 56 => self.castling &= !BLACK_QUEEN_SIDE,
            Piece::BlackRook if from == 63 => self.castling &= !BLACK_KING_SIDE,
            _ => {}
        }
        match to {
            0 => self.castling &= !WHITE_QUEEN_SIDE,
            7 => self.castling &= !WHITE_KING_SIDE,
            56 => self.castling &= !BLACK_QUEEN_SIDE,
            63 => self.castling &= !BLACK_KING_SIDE,
            _ => {}
        }

        self.ep_square = if piece == Piece::WhitePawn && to == from + 16 {
            Some((from + 8) as u8)
        } else if piece == Piece::BlackPawn && from == to + 16 {
            Some((from - 8) as u8)
        } else {
            None
        };
        if matches!(piece, Piece::WhitePawn | Piece::BlackPawn) || captured {
            self.halfmove_clock = 0;
        } else {
            self.halfmove_clock = self.halfmove_clock.saturating_add(1);
        }
        if !self.white_to_move {
            self.fullmove_number = self.fullmove_number.saturating_add(1);
        }
        self.white_to_move = !self.white_to_move;
    }
}

fn square_index(file: i8, rank: i8) -> Option<usize> {
    if (0..8).contains(&file) && (0..8).contains(&rank) {
        Some((rank * 8 + file) as usize)
    } else {
        None
    }
}

const KNIGHT_STEPS: [(i8, i8); 8] = [
    (-2, -1),
    (-2, 1),
    (-1, -2),
    (-1, 2),
    (1, -2),
    (1, 2),
    (2, -1),
    (2, 1),
];
const KING_STEPS: [(i8, i8); 8] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
];
const BISHOP_DIRECTIONS: [(i8, i8); 4] = [(-1, -1), (-1, 1), (1, -1), (1, 1)];
const ROOK_DIRECTIONS: [(i8, i8); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
const QUEEN_DIRECTIONS: [(i8, i8); 8] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
];

/// Plugin interface for an engine built on ZeroChessNet.
pub trait ZcnEngine {
    fn name(&self) -> &str {
        "ZeroChessNet"
    }

    fn author(&self) -> &str {
        "ZeroChessNet contributors"
    }

    fn new_game(&mut self) {}

    fn search(&mut self, board: &Board, eval: &ZcnEvaluator, ctx: &SearchContext) -> Move;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contains(board: &Board, uci: &str) -> bool {
        board
            .legal_moves()
            .iter()
            .any(|mv| mv.to_uci() == uci)
    }

    #[test]
    fn starting_position_has_twenty_legal_moves() {
        let board = Board::default();
        assert_eq!(board.legal_moves().len(), 20);

        fn perft(board: &Board, depth: u8) -> u64 {
            if depth == 0 {
                return 1;
            }
            board
                .legal_moves()
                .into_iter()
                .map(|mv| {
                    let mut next = board.clone();
                    next.apply_unchecked(mv);
                    perft(&next, depth - 1)
                })
                .sum()
        }
        assert_eq!(perft(&board, 2), 400);
    }

    #[test]
    fn rejects_moves_that_leave_king_in_check() {
        let board = Board::from_fen("4r1k1/8/8/8/8/8/4R3/4K3 w - - 0 1").unwrap();
        assert!(!contains(&board, "e2d2"));
        assert!(contains(&board, "e2e8"));
    }

    #[test]
    fn generates_castling_en_passant_and_promotion() {
        let castle = Board::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap();
        assert!(contains(&castle, "e1g1"));
        assert!(contains(&castle, "e1c1"));

        let ep = Board::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1").unwrap();
        assert!(contains(&ep, "e5d6"));

        let promotion = Board::from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        for suffix in ["q", "r", "b", "n"] {
            assert!(contains(&promotion, &format!("a7a8{suffix}")));
        }
    }

    #[test]
    fn move_application_updates_state_and_round_trips_fen() {
        let mut board = Board::default();
        board
            .try_apply_move(Move::from_uci("e2e4").unwrap())
            .unwrap();
        assert_eq!(board.to_fen(), "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
        assert!(board.try_apply_move(Move::from_uci("e7e5").unwrap()).is_ok());
        assert_eq!(board.fullmove_number(), 2);
    }

    #[test]
    fn rejects_invalid_fen_and_uci_moves() {
        assert!(Board::from_fen("8/8/8/8/8/8/8/8 w - - 0 1").is_ok());
        assert!(Board::from_fen("9/8/8/8/8/8/8/8 w - - 0 1").is_err());
        assert!(Move::from_uci("e2e4x").is_none());
        assert!(Move::from_uci("i2e4").is_none());
    }
}
