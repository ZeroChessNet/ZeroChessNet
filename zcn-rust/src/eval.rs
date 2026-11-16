use crate::Board;
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub struct ZcnEvaluator {
    nnue: Option<TznueNetwork>,
    loaded_path: Option<String>,
}

struct TznueNetwork {
    ft_weights: Vec<f32>,  // 768 * 256
    ft_biases: Vec<f32>,   // 256
    out_weights: Vec<f32>, // 512
    out_bias: f32,
}

impl Default for ZcnEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl ZcnEvaluator {
    pub fn new() -> Self {
        let mut eval = Self { nnue: None, loaded_path: None };
        if Path::new("titan-zero.nnue").exists() {
            let _ = eval.load_nnue("titan-zero.nnue");
        }
        eval
    }

    pub fn load_nnue(&mut self, path: &str) -> Result<(), String> {
        let mut file = File::open(path).map_err(|e| format!("Cannot open '{}': {}", path, e))?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf).map_err(|e| format!("Read error: {}", e))?;

        if buf.len() < 16 || &buf[0..8] != b"TZNUE001" {
            return Err("Invalid NNUE file: missing TZNUE001 magic header".to_string());
        }

        let float_bytes = &buf[12..];
        let expected_floats = 768 * 256 + 256 + 512 + 1;
        if float_bytes.len() < expected_floats * 4 {
            return Err(format!("Truncated TZNUE001 file ({} bytes)", buf.len()));
        }

        let mut floats = Vec::with_capacity(expected_floats);
        for chunk in float_bytes.chunks_exact(4).take(expected_floats) {
            floats.push(f32::from_le_bytes([chunk, chunk, chunk, chunk]));
        }

        let mut offset = 0;
        let ft_weights = floats[offset..offset + 768 * 256].to_vec();
        offset += 768 * 256;
        let ft_biases = floats[offset..offset + 256].to_vec();
        offset += 256;
        let out_weights = floats[offset..offset + 512].to_vec();
        offset += 512;
        let out_bias = floats[offset];

        self.nnue = Some(TznueNetwork { ft_weights, ft_biases, out_weights, out_bias });
        self.loaded_path = Some(path.to_string());
        Ok(())
    }

    pub fn loaded_file(&self) -> Option<&str> {
        self.loaded_path.as_deref()
    }

    pub fn evaluate(&self, board: &Board) -> i32 {
        if let Some(net) = &self.nnue {
            let mut acc_w = net.ft_biases.clone();
            let mut acc_b = net.ft_biases.clone();

            for (sq, cell) in board.squares.iter().enumerate() {
                if let Some(piece) = cell {
                    let feat_w = (piece as usize) * 64 + sq;
                    let feat_b = (piece.mirror_color() as usize) * 64 + (sq ^ 56);

                    let w_row = &net.ft_weights[feat_w * 256..(feat_w + 1) * 256];
                    let b_row = &net.ft_weights[feat_b * 256..(feat_b + 1) * 256];

                    for i in 0..256 {
                        acc_w[i] += w_row[i];
                        acc_b[i] += b_row[i];
                    }
                }
            }

            let (stm, nstm) = if board.white_to_move { (&acc_w, &acc_b) } else { (&acc_b, &acc_w) };
            let mut sum = net.out_bias;
            for i in 0..256 {
                sum += stm[i].clamp(0.0, 1.0) * net.out_weights[i];
                sum += nstm[i].clamp(0.0, 1.0) * net.out_weights[256 + i];
            }
            sum.round() as i32
        } else {
            self.evaluate_pesto_fallback(board)
        }
    }

    fn evaluate_pesto_fallback(&self, board: &Board) -> i32 {
        const VALS: [i32; 6] =;
        let mut score = 0;
        for cell in board.squares.iter().flatten() {
            let idx = (*cell as usize) % 6;
            if cell.is_white() { score += VALS[idx]; } else { score -= VALS[idx]; }
        }
        if board.white_to_move { score } else { -score }
    }
}
