use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct SearchContext {
    pub start_time: Instant,
    pub soft_limit: Duration,
    pub hard_limit: Duration,
    pub max_depth: u8,
    pub stop_flag: Arc<AtomicBool>,
}

impl SearchContext {
    pub fn from_uci_go(line: &str, white_to_move: bool) -> Self {
        let mut wtime = None::<u64>;
        let mut btime = None::<u64>;
        let mut winc = 0u64;
        let mut binc = 0u64;
        let mut movetime = None::<u64>;
        let mut depth = 64u8;

        let mut it = line.split_whitespace().skip(1);
        while let Some(tok) = it.next() {
            match tok {
                "wtime" => wtime = it.next().and_then(|v| v.parse().ok()),
                "btime" => btime = it.next().and_then(|v| v.parse().ok()),
                "winc" => winc = it.next().and_then(|v| v.parse().ok()).unwrap_or(0),
                "binc" => binc = it.next().and_then(|v| v.parse().ok()).unwrap_or(0),
                "movetime" => movetime = it.next().and_then(|v| v.parse().ok()),
                "depth" => depth = it.next().and_then(|v| v.parse().ok()).unwrap_or(64),
                _ => {}
            }
        }

        let (soft_ms, hard_ms) = if let Some(mt) = movetime {
            let safe = mt.saturating_sub(15).max(1);
            (safe, safe)
        } else {
            let (time_left, inc) = if white_to_move { (wtime, winc) } else { (btime, binc) };
            if let Some(tl) = time_left {
                let hard = tl.saturating_sub(50).max(5);
                let soft = (tl / 25 + inc.saturating_mul(7) / 10)
                    .min(hard)
                    .max(5);
                (soft, hard)
            } else {
                (30_000, 30_000)
            }
        };

        Self {
            start_time: Instant::now(),
            soft_limit: Duration::from_millis(soft_ms),
            hard_limit: Duration::from_millis(hard_ms),
            max_depth: depth,
            stop_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn should_stop(&self) -> bool {
        self.stop_flag.load(Ordering::Relaxed) || self.start_time.elapsed() >= self.hard_limit
    }

    pub fn soft_expired(&self) -> bool {
        self.stop_flag.load(Ordering::Relaxed) || self.start_time.elapsed() >= self.soft_limit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_side_specific_clock_and_depth() {
        let context =
            SearchContext::from_uci_go("go wtime 60000 btime 30000 winc 1000 depth 12", true);
        assert_eq!(context.max_depth, 12);
        assert_eq!(context.soft_limit, Duration::from_millis(3100));
        assert_eq!(context.hard_limit, Duration::from_millis(59950));
    }

    #[test]
    fn movetime_is_a_hard_upper_bound() {
        let context = SearchContext::from_uci_go("go movetime 100", false);
        assert_eq!(context.soft_limit, Duration::from_millis(85));
        assert_eq!(context.hard_limit, Duration::from_millis(85));
    }
}
