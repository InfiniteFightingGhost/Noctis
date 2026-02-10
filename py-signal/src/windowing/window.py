import src.types as types
import src.windowing.window as window
import src.windowing.config as config
import src.windowing.state as state


def window_batch(
    signals: types.MultiSignal,
    start_ts: int,
    cfg: config.WindowConfig,
) -> list[types.Window]:
    # ) -> None:
    if not signals:
        raise ValueError("No signals provided")

    lengths = {len(sig) for sig in signals.values()}
    if len(lengths) != 1:
        raise ValueError("All channells must have equal length")

    total_samples = lengths.pop()
    if total_samples < cfg.size_samples:
        return []
    windows: list[types.Window] = []

    for start in range(0, total_samples - cfg.size_samples + 1, cfg.stride_samples):
        end = start + cfg.size_samples

        window_data = {name: sig[start:end] for name, sig in signals.items()}

        win_start_ts = start_ts + start
        win_end_ts = start_ts + end

        windows.append(
            types.Window(start_ts=win_start_ts, end_ts=win_end_ts, data=window_data)
        )

    return windows


def window_incremental(
    state: state.WindowState,
    new_samples: types.MultiSignal,
    start_ts: int,
    cfg: config.WindowConfig,
    # ) -> tuple[list[Window], WindowState]:
) -> None:
    pass
