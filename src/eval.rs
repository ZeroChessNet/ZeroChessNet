use crate::Board;
use std::fs;
use std::path::Path;

const HEADER_BYTES: usize = 12;
const FEATURE_WEIGHT_COUNT: usize = 768 * 256;
const FEATURE_BIAS_COUNT: usize = 256;
const OUTPUT_WEIGHT_COUNT: usize = 512;
const NETWORK_FLOAT_COUNT: usize =
    FEATURE_WEIGHT_COUNT + FEATURE_BIAS_COUNT + OUTPUT_WEIGHT_COUNT + 1;

#[derive(Clone)]
pub(crate) struct TznueNetwork {
    pub(crate) ft_weights: Vec<f32>,
    pub(crate) ft_biases: Vec<f32>,
    pub(crate) out_weights: Vec<f32>,
    pub(crate) out_bias: f32,
}

pub struct ZcnEvaluator {
    pub(crate) nnue: Option<TznueNetwork>,
    loaded_path: Option<String>,
}

impl Default for ZcnEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl ZcnEvaluator {
    pub fn new() -> Self {
        let mut evaluator = Self {
            nnue: None,
            loaded_path: None,
        };
        if Path::new("titan-zero.nnue").is_file() {
            if let Err(error) = evaluator.load_nnue("titan-zero.nnue") {
                eprintln!("ZeroChessNet: failed to load titan-zero.nnue: {error}");
            }
        }
        evaluator
    }

    pub fn load_nnue(&mut self, path: impl AsRef<Path>) -> Result<(), String> {
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(|error| {
            format!("Cannot read '{}': {error}", path.display())
        })?;
        let expected_len = HEADER_BYTES + NETWORK_FLOAT_COUNT * 4;
        if bytes.len() < HEADER_BYTES || &bytes[..8] != b"TZNUE001" {
            return Err("Invalid NNUE file: missing TZNUE001 header".to_owned());
        }
        if bytes.len() != expected_len {
            return Err(format!(
                "Invalid TZNUE001 file size: expected {expected_len} bytes, got {}",
                bytes.len()
            ));
        }

        let mut floats = Vec::with_capacity(NETWORK_FLOAT_COUNT);
        for chunk in bytes[HEADER_BYTES..].chunks_exact(4) {
            let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            if !value.is_finite() {
                return Err("Invalid TZNUE001 file: weights must be finite".to_owned());
            }
            floats.push(value);
        }

        let mut offset = 0;
        let ft_weights = floats[offset..offset + FEATURE_WEIGHT_COUNT].to_vec();
        offset += FEATURE_WEIGHT_COUNT;
        let ft_biases = floats[offset..offset + FEATURE_BIAS_COUNT].to_vec();
        offset += FEATURE_BIAS_COUNT;
        let out_weights = floats[offset..offset + OUTPUT_WEIGHT_COUNT].to_vec();
        offset += OUTPUT_WEIGHT_COUNT;
        let out_bias = floats[offset];

        self.nnue = Some(TznueNetwork {
            ft_weights,
            ft_biases,
            out_weights,
            out_bias,
        });
        self.loaded_path = Some(path.to_string_lossy().into_owned());
        Ok(())
    }

    pub fn loaded_file(&self) -> Option<&str> {
        self.loaded_path.as_deref()
    }

    pub fn evaluate(&self, board: &Board) -> i32 {
        if let Some(net) = &self.nnue {
            let mut accumulator_white = net.ft_biases.clone();
            let mut accumulator_black = net.ft_biases.clone();

            for (square, cell) in board.squares.iter().enumerate() {
                if let Some(piece) = cell {
                    let feature_white = (*piece as usize) * 64 + square;
                    let feature_black =
                        (piece.mirror_color() as usize) * 64 + (square ^ 56);
                    let white_row =
                        &net.ft_weights[feature_white * 256..(feature_white + 1) * 256];
                    let black_row =
                        &net.ft_weights[feature_black * 256..(feature_black + 1) * 256];
                    for index in 0..256 {
                        accumulator_white[index] += white_row[index];
                        accumulator_black[index] += black_row[index];
                    }
                }
            }

            let (side_to_move, other_side) = if board.white_to_move() {
                (&accumulator_white, &accumulator_black)
            } else {
                (&accumulator_black, &accumulator_white)
            };
            let mut sum = net.out_bias;
            for index in 0..256 {
                sum += side_to_move[index].clamp(0.0, 1.0) * net.out_weights[index];
                sum += other_side[index].clamp(0.0, 1.0) * net.out_weights[256 + index];
            }
            sum.round() as i32
        } else {
            self.evaluate_pesto_fallback(board)
        }
    }

    fn evaluate_pesto_fallback(&self, board: &Board) -> i32 {
        const VALUES: [i32; 6] = [100, 320, 330, 500, 900, 0];
        let mut score = 0;
        for piece in board.squares.iter().flatten() {
            let value = VALUES[(*piece as usize) % 6];
            if piece.is_white() {
                score += value;
            } else {
                score -= value;
            }
        }
        if board.white_to_move() {
            score
        } else {
            -score
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_evaluation_is_relative_to_side_to_move() {
        let white_to_move = Board::from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1").unwrap();
        let black_to_move = Board::from_fen("4k3/8/8/8/8/8/4P3/4K3 b - - 0 1").unwrap();
        let evaluator = ZcnEvaluator {
            nnue: None,
            loaded_path: None,
        };
        assert_eq!(evaluator.evaluate(&white_to_move), 100);
        assert_eq!(evaluator.evaluate(&black_to_move), -100);
    }
}
