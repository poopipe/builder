"""fit road kit pieces along a polyline (radius-based fillets)"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import atan2, ceil, cos, degrees, hypot, radians, sin, tan

from builder.heightfield.road_sections.kit_widths import PieceSnapPoints
from builder.heightfield.road_sections.part_catalog import (
    CornerPiece,
    RoadKit,
    straight_stem,
)
from builder.heightfield.road_sections.shp_import import (
    simplify_eps_m,
    simplify_polyline,
)

# prefer a single corner when its angle is this close to the turn
corner_angle_tol: float = 12.0
# leftover turn smaller than this is not stacked further
corner_stack_min_deg: float = 1.0
# metres reserved so fillet tangents do not hit the vertex
fillet_leg_margin_m: float = 1.0
corner_radius_slack_m: float = 0.5


@dataclass(frozen=True)
class FittedPiece:
    """one placed road piece in polyline xz space"""

    stem: str
    x: float
    z: float
    yaw_deg: float
    roll_deg: float = 0.0


def dedupe_pts(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """drop consecutive duplicate vertices"""
    out: list[tuple[float, float]] = []
    point: tuple[float, float]
    for point in pts:
        if not out or (
            abs(point[0] - out[-1][0]) > 1e-6 or abs(point[1] - out[-1][1]) > 1e-6
        ):
            out.append(point)
    return out


def chain_length(pts: list[tuple[float, float]]) -> float:
    """total length of a polyline"""
    total: float = 0.0
    a: tuple[float, float]
    b: tuple[float, float]
    for a, b in zip(pts, pts[1:]):
        total += hypot(b[0] - a[0], b[1] - a[1])
    return total


def angle_diff_deg(a: float, b: float) -> float:
    """signed delta b-a in (-180, 180], positive = clockwise"""
    d: float = (b - a + 180.0) % 360.0 - 180.0
    if d <= -180.0:
        d += 360.0
    return d


def heading(a: tuple[float, float], b: tuple[float, float]) -> float:
    """yaw degrees; atan2(dx, dz) matches gtt map xy heading"""
    return degrees(atan2(b[0] - a[0], b[1] - a[1])) % 360.0


def point_at(
    pts: list[tuple[float, float]], dist: float
) -> tuple[float, float, float]:
    """point and heading at distance along the polyline"""
    if len(pts) < 2:
        x: float
        z: float
        x, z = pts[0]
        return x, z, 0.0
    remaining: float = max(0.0, dist)
    a: tuple[float, float]
    b: tuple[float, float]
    for a, b in zip(pts, pts[1:]):
        seg: float = hypot(b[0] - a[0], b[1] - a[1])
        if seg < 1e-9:
            continue
        if remaining <= seg:
            t: float = remaining / seg
            return (
                a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t,
                heading(a, b),
            )
        remaining -= seg
    a, b = pts[-2], pts[-1]
    return b[0], b[1], heading(a, b)


def closest_on_polyline(
    pts: list[tuple[float, float]], x: float, z: float
) -> tuple[float, float, float]:
    """closest point and tangent yaw on an open polyline"""
    if not pts:
        return x, z, 0.0
    if len(pts) == 1:
        return pts[0][0], pts[0][1], 0.0
    best_x: float = pts[0][0]
    best_z: float = pts[0][1]
    best_yaw: float = heading(pts[0], pts[1])
    best_d2: float = (x - best_x) * (x - best_x) + (z - best_z) * (z - best_z)
    a: tuple[float, float]
    b: tuple[float, float]
    for a, b in zip(pts, pts[1:]):
        dx: float = b[0] - a[0]
        dz: float = b[1] - a[1]
        seg2: float = dx * dx + dz * dz
        if seg2 < 1e-18:
            continue
        t: float = ((x - a[0]) * dx + (z - a[1]) * dz) / seg2
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        px: float = a[0] + dx * t
        pz: float = a[1] + dz * t
        d2: float = (x - px) * (x - px) + (z - pz) * (z - pz)
        if d2 < best_d2:
            best_d2 = d2
            best_x = px
            best_z = pz
            best_yaw = heading(a, b)
    return best_x, best_z, best_yaw


def vertex_distances(pts: list[tuple[float, float]]) -> list[float]:
    """cumulative distance at each vertex"""
    dists: list[float] = [0.0]
    a: tuple[float, float]
    b: tuple[float, float]
    for a, b in zip(pts, pts[1:]):
        dists.append(dists[-1] + hypot(b[0] - a[0], b[1] - a[1]))
    return dists


def vertex_turns(pts: list[tuple[float, float]]) -> list[float]:
    """signed turn at each interior vertex"""
    turns: list[float] = [0.0] * len(pts)
    i: int
    for i in range(1, len(pts) - 1):
        h_in: float = heading(pts[i - 1], pts[i])
        h_out: float = heading(pts[i], pts[i + 1])
        turns[i] = angle_diff_deg(h_in, h_out)
    return turns


def max_fillet_radius_m(
    turn_deg: float,
    prev_len: float,
    next_len: float,
    *,
    margin_m: float = fillet_leg_margin_m,
) -> float:
    """largest circular-fillet radius that fits both legs for this turn"""
    half: float = abs(turn_deg) * 0.5
    if half < 1e-3:
        return 0.0
    t: float = tan(radians(half))
    if t < 1e-9:
        return 0.0
    usable: float = min(prev_len, next_len) - margin_m
    if usable <= 0.0:
        return 0.0
    return usable / t


def fillet_leg_m(corner: CornerPiece) -> float:
    """distance from vertex along each leg to the circular-arc tangent point"""
    a: float = radians(max(corner.angle_deg, 1e-3))
    return corner.radius_m * tan(a * 0.5)


def stack_fillet_leg_m(stack: list[tuple[CornerPiece, float]]) -> float:
    """approx leg length reserved before/after the vertex for a corner stack"""
    return sum(fillet_leg_m(c) for c, _ in stack)


def corner_fits_legs(
    corner: CornerPiece,
    prev_len: float,
    next_len: float,
    r_max: float,
) -> bool:
    """true when corner radius and fillet legs fit this bend"""
    if corner.radius_m > r_max + corner_radius_slack_m:
        return False
    leg: float = fillet_leg_m(corner)
    return leg <= prev_len - fillet_leg_margin_m and leg <= next_len - fillet_leg_margin_m


def decompose_turn(
    turn_deg: float,
    kit: RoadKit,
    prev_len: float,
    next_len: float,
) -> list[tuple[CornerPiece, float]]:
    """pick corner pieces by fillet radius fit; prefer angle match, then larger r"""
    if not kit.corners:
        return []
    if abs(turn_deg) < corner_stack_min_deg:
        return []
    r_max: float = max_fillet_radius_m(turn_deg, prev_len, next_len)
    if r_max <= 0.0:
        return []

    target: float = abs(turn_deg)
    roll: float = 0.0 if turn_deg >= 0.0 else 180.0
    pool: list[CornerPiece] = [
        corner
        for corner in kit.corners
        if corner_fits_legs(corner, prev_len, next_len, r_max)
    ]
    if not pool:
        return []

    # single piece: best angle match, then largest radius (gentlest fit)
    singles: list[tuple[float, float, CornerPiece]] = sorted(
        (
            (abs(c.angle_deg - target), -c.radius_m, c)
            for c in pool
        ),
    )
    best_err: float
    best: CornerPiece
    best_err, _, best = singles[0]
    if best_err <= corner_angle_tol:
        return [(best, roll)]

    # stack smaller angles that still fit the same r_max / legs
    remaining: float = target
    picked: list[tuple[CornerPiece, float]] = []
    budget_prev: float = prev_len
    budget_next: float = next_len
    guard: int = 0
    while remaining >= corner_stack_min_deg and guard < 12:
        guard += 1
        scored: list[tuple[float, float, CornerPiece]] = []
        corner: CornerPiece
        for corner in pool:
            if corner.angle_deg > remaining + corner_angle_tol:
                continue
            if not corner_fits_legs(corner, budget_prev, budget_next, r_max):
                continue
            if picked and corner.arc_length_m > max(budget_prev, budget_next):
                continue
            scored.append(
                (abs(corner.angle_deg - remaining), -corner.radius_m, corner)
            )
        if not scored:
            break
        scored.sort()
        found: CornerPiece = scored[0][2]
        picked.append((found, roll))
        remaining -= found.angle_deg
        leg: float = fillet_leg_m(found)
        budget_prev = max(fillet_leg_margin_m, budget_prev - leg)
        budget_next = max(fillet_leg_margin_m, budget_next - leg)
    return picked


# max total overshoot / joint overlap budget as a fraction of shortest straight
straight_overlap_fraction: float = 0.5


def choose_straight_lengths(
    span_m: float, straights_m: tuple[int, ...]
) -> list[int]:
    """kit lengths that meet or slightly exceed span_m

    prefer sum in [ceil(span), floor(span + 0.5*shortest)], least overshoot,
    then fewest pieces, then longer pieces. fall back to max sum ≤ floor(span)
    """
    if not straights_m or span_m < 0.5:
        return []
    coins: tuple[int, ...] = tuple(sorted(set(straights_m), reverse=True))
    shortest: int = coins[-1]
    max_overlap: float = straight_overlap_fraction * float(shortest)
    min_s: int = int(ceil(span_m - 1e-9))
    max_s: int = int(span_m + max_overlap + 1e-9)
    if float(shortest) <= span_m + max_overlap + 1e-9 and max_s < shortest:
        return [shortest]

    dp_limit: int = max(max_s, int(span_m + 1e-9), shortest)
    inf: int = dp_limit + 1
    best_count: list[int] = [inf] * (dp_limit + 1)
    best_count[0] = 0
    s: int
    for s in range(dp_limit + 1):
        if best_count[s] >= inf:
            continue
        coin: int
        for coin in coins:
            nxt: int = s + coin
            if nxt > dp_limit:
                continue
            new_count: int = best_count[s] + 1
            if new_count < best_count[nxt]:
                best_count[nxt] = new_count

    def reconstruct(best_s: int) -> list[int]:
        lengths: list[int] = []
        cur: int = best_s
        while cur > 0:
            chosen: int | None = None
            coin: int
            for coin in coins:
                prev: int = cur - coin
                if prev >= 0 and best_count[prev] == best_count[cur] - 1:
                    chosen = coin
                    break
            if chosen is None:
                break
            lengths.append(chosen)
            cur -= chosen
        return lengths

    # least overshoot in [min_s, max_s]
    s = max(min_s, 0)
    while s <= max_s:
        if s <= dp_limit and best_count[s] < inf:
            return reconstruct(s)
        s += 1

    # fall back: max sum ≤ floor(span)
    floor_s: int = int(span_m + 1e-9)
    for s in range(min(floor_s, dp_limit), -1, -1):
        if best_count[s] < inf:
            return reconstruct(s)
    if float(shortest) <= span_m + max_overlap + 1e-9:
        return [shortest]
    return []


def fill_straights(
    pts: list[tuple[float, float]],
    kit: RoadKit,
    start_d: float,
    end_d: float,
    yaw: float,
    *,
    length_for_stem: Callable[[str], float] | None = None,
) -> list[FittedPiece]:
    """place straights spanning [start_d, end_d] with slight overlap if needed

    chooses declared stems that meet/slightly exceed the span, then spaces
    centers using measured stem lengths when length_for_stem is provided.
    """
    pieces: list[FittedPiece] = []
    if not kit.straights_m:
        return pieces
    span: float = end_d - start_d
    if span < 0.5:
        return pieces
    lengths: list[int] = choose_straight_lengths(span, kit.straights_m)
    if not lengths:
        return pieces

    stems: list[str] = [straight_stem(kit, length) for length in lengths]
    measured: list[float] = []
    stem: str
    length: int
    for stem, length in zip(stems, lengths, strict=True):
        if length_for_stem is not None:
            measured.append(max(float(length_for_stem(stem)), 0.5))
        else:
            measured.append(float(length))

    total: float = float(sum(measured))
    n: int = len(measured)
    centers: list[float] = []
    if n == 1:
        centers.append(0.5 * (start_d + end_d))
    elif total <= span + 1e-9:
        cursor: float = start_d
        meas: float
        for meas in measured:
            centers.append(cursor + meas * 0.5)
            cursor += meas
    else:
        overshoot: float = total - span
        joint_overlap: float = overshoot / float(n - 1)
        cursor = start_d + measured[0] * 0.5
        centers.append(cursor)
        i: int
        for i in range(n - 1):
            step: float = (
                measured[i] * 0.5 + measured[i + 1] * 0.5 - joint_overlap
            )
            cursor += step
            centers.append(cursor)

    i = 0
    while i < n:
        x: float
        z: float
        x, z, _ = point_at(pts, centers[i])
        pieces.append(
            FittedPiece(
                stem=stems[i],
                x=x,
                z=z,
                yaw_deg=yaw,
            )
        )
        i += 1
    return pieces


def heading_unit(yaw_deg: float) -> tuple[float, float]:
    """unit forward (x, z) for a yaw"""
    h: float = radians(yaw_deg)
    return sin(h), cos(h)


def rotate_cw(vx: float, vz: float, deg: float) -> tuple[float, float]:
    """rotate a vector clockwise about +Y by deg"""
    r: float = radians(deg)
    c: float = cos(r)
    s: float = sin(r)
    return vx * c + vz * s, -vx * s + vz * c


def corner_placement_yaw_deg(
    entry_yaw_deg: float,
    angle_deg: float,
    *,
    clockwise: bool,
) -> float:
    """yaw for a cw corner mesh: entry heading, or y-spin reverse for ccw

    reverse offset is 180 - angle so endpoints land on the ccw fillet with +Y up
    (a flat +180 only matches a 180° piece).
    """
    if clockwise:
        return entry_yaw_deg % 360.0
    return (entry_yaw_deg + 180.0 - angle_deg) % 360.0


def corner_mesh_travel_entry_local(
    corner: CornerPiece, *, clockwise: bool
) -> tuple[float, float]:
    """mesh-local xz that maps to travel entry under placement yaw"""
    # cw: mesh begin. ccw y-spin: mesh end runs at travel entry
    t: float = 0.0 if clockwise else 1.0
    return corner_local_centerline_xz(corner, t)


def corner_snap_origin_delta_local(
    corner: CornerPiece,
    snaps: PieceSnapPoints,
    *,
    clockwise: bool,
) -> tuple[float, float]:
    """mid-arc local → snap/mesh origin: ideal_entry - snap_entry

    placement at mid-arc M becomes P = M + R(yaw)*delta so travel entry lands
    on the planned fillet entry when the real mesh origin is used.
    """
    ideal_x: float
    ideal_z: float
    ideal_x, ideal_z = corner_mesh_travel_entry_local(
        corner, clockwise=clockwise
    )
    snap_x: float
    snap_z: float
    if clockwise:
        snap_x, snap_z = snaps.begin_mid_xz()
    else:
        snap_x, snap_z = snaps.end_mid_xz()
    return ideal_x - snap_x, ideal_z - snap_z


def apply_yaw_xz(
    origin_x: float,
    origin_z: float,
    yaw_deg: float,
    local_x: float,
    local_z: float,
) -> tuple[float, float]:
    """world xz = origin + Ry(yaw) * local xz"""
    h: float = radians(yaw_deg)
    c: float = cos(h)
    s: float = sin(h)
    return (
        origin_x + local_x * c + local_z * s,
        origin_z - local_x * s + local_z * c,
    )


def corner_local_centerline_xz(
    corner: CornerPiece,
    t: float,
) -> tuple[float, float]:
    """cw mesh centerline at t in [0, 1]; origin at mid-arc, entry at t=0"""
    angle_deg: float = max(float(corner.angle_deg), 1e-3)
    radius: float = float(corner.radius_m)
    half: float = radians(angle_deg * 0.5)
    mid_x: float = radius * (1.0 - cos(half))
    mid_z: float = radius * sin(half)
    theta: float = radians(angle_deg * t)
    px: float = radius - radius * cos(theta) - mid_x
    pz: float = radius * sin(theta) - mid_z
    return px, pz


def place_corner_stack_arc(
    entry_x: float,
    entry_z: float,
    yaw_in: float,
    stack: list[tuple[CornerPiece, float]],
    *,
    snaps_for_stem: Callable[[str], PieceSnapPoints | None] | None = None,
) -> tuple[list[FittedPiece], float, float, float]:
    """place corner prefabs along arcs; returns pieces and exit x, z, yaw

    stack pairs are (corner, roll_flag) with roll_flag 0=cw / 180=ccw for planning;
    stored yaw is y-spin for display/export; roll 0/180 is chirality only (not
    applied as mesh roll). position uses snap/mesh origin when snaps exist.
    """
    pieces: list[FittedPiece] = []
    ex: float = entry_x
    ez: float = entry_z
    h: float = yaw_in
    corner: CornerPiece
    roll: float
    for corner, roll in stack:
        clockwise: bool = roll < 90.0
        sign: float = 1.0 if clockwise else -1.0
        radius: float = corner.radius_m
        ux: float
        uz: float
        ux, uz = heading_unit(h)
        cx: float
        cz: float
        if clockwise:
            cx, cz = ex + radius * uz, ez - radius * ux
        else:
            cx, cz = ex - radius * uz, ez + radius * ux
        half: float = sign * corner.angle_deg * 0.5
        mx: float
        mz: float
        mx, mz = rotate_cw(ex - cx, ez - cz, half)
        mx, mz = cx + mx, cz + mz
        yaw_deg: float = corner_placement_yaw_deg(
            h, corner.angle_deg, clockwise=clockwise
        )
        px: float = mx
        pz: float = mz
        if snaps_for_stem is not None:
            snaps: PieceSnapPoints | None = snaps_for_stem(corner.stem)
            if snaps is not None:
                dx: float
                dz: float
                dx, dz = corner_snap_origin_delta_local(
                    corner, snaps, clockwise=clockwise
                )
                ox: float
                oz: float
                ox, oz = apply_yaw_xz(0.0, 0.0, yaw_deg, dx, dz)
                px = mx + ox
                pz = mz + oz
        pieces.append(
            FittedPiece(
                stem=corner.stem,
                x=px,
                z=pz,
                yaw_deg=yaw_deg,
                # chirality flag only; mesh transform / tb export keep roll 0
                roll_deg=0.0 if clockwise else 180.0,
            )
        )
        full: float = sign * corner.angle_deg
        qx: float
        qz: float
        qx, qz = rotate_cw(ex - cx, ez - cz, full)
        ex, ez = cx + qx, cz + qz
        h = (h + full) % 360.0
    return pieces, ex, ez, h


def fill_straights_between(
    start: tuple[float, float],
    end: tuple[float, float],
    kit: RoadKit,
    *,
    length_for_stem: Callable[[str], float] | None = None,
) -> tuple[list[FittedPiece], tuple[float, float]]:
    """pack straights along the chord; returns pieces and where the pack ends"""
    span: float = hypot(end[0] - start[0], end[1] - start[1])
    if span < 0.5:
        return [], start
    yaw: float = heading(start, end)
    pieces: list[FittedPiece] = fill_straights(
        [start, end],
        kit,
        0.0,
        span,
        yaw,
        length_for_stem=length_for_stem,
    )
    if not pieces:
        return [], start
    measured_total: float
    if length_for_stem is not None:
        measured_total = 0.0
        for piece in pieces:
            measured_total += max(float(length_for_stem(piece.stem)), 0.5)
    else:
        declared: list[int] = choose_straight_lengths(span, kit.straights_m)
        measured_total = float(sum(declared))
    ux: float
    uz: float
    ux, uz = heading_unit(yaw)
    if len(pieces) == 1:
        mid_x: float = 0.5 * (start[0] + end[0])
        mid_z: float = 0.5 * (start[1] + end[1])
        half: float = measured_total * 0.5
        return pieces, (mid_x + half * ux, mid_z + half * uz)
    if measured_total <= span + 1e-9:
        # under-cover / exact: pack runs end-to-end from start
        return pieces, (
            start[0] + measured_total * ux,
            start[1] + measured_total * uz,
        )
    # overshoot: last piece ends on the target junction
    return pieces, end


def sample_fillet_arc(
    entry_x: float,
    entry_z: float,
    yaw_in: float,
    corner: CornerPiece,
    roll: float,
    samples: int,
) -> list[tuple[float, float]]:
    """sample a temporary centerline fillet (excluding the entry point)"""
    clockwise: bool = roll < 90.0
    sign: float = 1.0 if clockwise else -1.0
    radius: float = corner.radius_m
    ux: float
    uz: float
    ux, uz = heading_unit(yaw_in)
    cx: float
    cz: float
    if clockwise:
        cx, cz = entry_x + radius * uz, entry_z - radius * ux
    else:
        cx, cz = entry_x - radius * uz, entry_z + radius * ux
    out: list[tuple[float, float]] = []
    i: int
    for i in range(1, samples + 1):
        t: float = float(i) / float(samples)
        ang: float = sign * corner.angle_deg * t
        qx: float
        qz: float
        qx, qz = rotate_cw(entry_x - cx, entry_z - cz, ang)
        out.append((cx + qx, cz + qz))
    return out


def build_filleted_centerline(
    pts: list[tuple[float, float]],
    plans: dict[int, tuple[list[tuple[CornerPiece, float]], float]],
    *,
    arc_samples: int = 8,
) -> list[tuple[float, float]]:
    """temporary filleted path; straight spans meet real arc entry/exit"""
    if len(pts) < 2:
        return list(pts)
    dists: list[float] = vertex_distances(pts)
    out: list[tuple[float, float]] = [pts[0]]
    i: int
    for i in range(len(pts) - 1):
        yaw_in: float = heading(pts[i], pts[i + 1])
        corner_at_end: tuple[list[tuple[CornerPiece, float]], float] | None = (
            plans.get(i + 1)
        )
        if corner_at_end is None:
            out.append(pts[i + 1])
            continue
        stack, leg = corner_at_end
        ex: float
        ez: float
        ex, ez, _ = point_at(pts, dists[i + 1] - leg)
        out.append((ex, ez))
        h: float = yaw_in
        cur_x: float = ex
        cur_z: float = ez
        corner: CornerPiece
        roll: float
        for corner, roll in stack:
            samples: list[tuple[float, float]] = sample_fillet_arc(
                cur_x, cur_z, h, corner, roll, arc_samples
            )
            out.extend(samples)
            cur_x, cur_z = samples[-1]
            sign: float = 1.0 if roll < 90.0 else -1.0
            h = (h + sign * corner.angle_deg) % 360.0
        # keep the real arc exit; do not snap back onto the chord
    return dedupe_pts(out)


def curve_merge_span_m(kit: RoadKit) -> float:
    """gaps shorter than this between fillets are merged (no straight)"""
    if kit.straights_m:
        return float(min(kit.straights_m))
    return 6.0


def gap_should_merge_curves(gap_m: float, kit: RoadKit) -> bool:
    """true only for short inter-fillet stubs — not long under-covered spans"""
    if gap_m < 0.5:
        return True
    merge_span: float = curve_merge_span_m(kit)
    if gap_m < merge_span:
        return True
    if not kit.straights_m:
        return gap_m < merge_span
    lengths: list[int] = choose_straight_lengths(gap_m, kit.straights_m)
    if not lengths:
        return gap_m < merge_span
    shortest: float = float(min(kit.straights_m))
    # lone shortest in a barely-longer gap (~6.9 m) leaves gaps either side
    if (
        len(lengths) == 1
        and float(lengths[0]) <= shortest + 1e-9
        and gap_m < shortest + 1.0
    ):
        return True
    return False


def fit_polyline_pieces(
    pts: list[tuple[float, float]],
    kit: RoadKit,
    *,
    min_length_m: float = 6.0,
    length_for_stem: Callable[[str], float] | None = None,
    snaps_for_stem: Callable[[str], PieceSnapPoints | None] | None = None,
) -> list[FittedPiece]:
    """fit straights between fillets; merge curves when the gap is too short

    length_for_stem / snaps_for_stem calibrate packing and corner origins to
    measured mesh frames when provided.
    """
    pts = simplify_polyline(dedupe_pts(pts), simplify_eps_m)
    if len(pts) < 2:
        return []
    total: float = chain_length(pts)
    if total < min_length_m:
        return []

    dists: list[float] = vertex_distances(pts)
    turns: list[float] = vertex_turns(pts)
    merge_span: float = curve_merge_span_m(kit)

    plans: dict[int, tuple[list[tuple[CornerPiece, float]], float]] = {}
    i: int
    for i in range(1, len(pts) - 1):
        prev_len: float = dists[i] - dists[i - 1]
        next_len: float = dists[i + 1] - dists[i]
        stack: list[tuple[CornerPiece, float]] = decompose_turn(
            turns[i], kit, prev_len, next_len
        )
        if not stack:
            continue
        leg: float = stack_fillet_leg_m(stack)
        plans[i] = (stack, leg)

    pieces: list[FittedPiece] = []
    pos: tuple[float, float] = pts[0]
    from_fillet: bool = False
    exit_yaw: float = heading(pts[0], pts[1]) if len(pts) >= 2 else 0.0

    for i in range(len(pts) - 1):
        yaw_in: float = heading(pts[i], pts[i + 1])
        corner_at_end: tuple[list[tuple[CornerPiece, float]], float] | None = (
            plans.get(i + 1)
        )
        if corner_at_end is None:
            filled: list[FittedPiece]
            pack_end: tuple[float, float]
            filled, pack_end = fill_straights_between(
                pos,
                pts[i + 1],
                kit,
                length_for_stem=length_for_stem,
            )
            pieces.extend(filled)
            pos = pack_end if filled else pts[i + 1]
            from_fillet = False
            exit_yaw = yaw_in
            continue

        stack, leg = corner_at_end
        entry_x: float
        entry_z: float
        entry_x, entry_z, _ = point_at(pts, dists[i + 1] - leg)
        entry: tuple[float, float] = (entry_x, entry_z)
        gap: float = hypot(entry[0] - pos[0], entry[1] - pos[1])

        corner_pieces: list[FittedPiece]
        exit_x: float
        exit_z: float
        if from_fillet and gap_should_merge_curves(gap, kit):
            # absorb short / stubby inter-fillet gap: continue arcs from exit
            corner_pieces, exit_x, exit_z, exit_yaw = place_corner_stack_arc(
                pos[0],
                pos[1],
                exit_yaw,
                stack,
                snaps_for_stem=snaps_for_stem,
            )
        else:
            filled, pack_end = fill_straights_between(
                pos,
                entry,
                kit,
                length_for_stem=length_for_stem,
            )
            pieces.extend(filled)
            # short under-cover: start the corner at the pack end (mild reshape)
            corner_x: float = entry_x
            corner_z: float = entry_z
            leftover: float = hypot(
                entry_x - pack_end[0], entry_z - pack_end[1]
            )
            if filled and 0.05 < leftover < merge_span:
                corner_x, corner_z = pack_end
            corner_pieces, exit_x, exit_z, exit_yaw = place_corner_stack_arc(
                corner_x,
                corner_z,
                yaw_in,
                stack,
                snaps_for_stem=snaps_for_stem,
            )
        pieces.extend(corner_pieces)
        pos = (exit_x, exit_z)
        from_fillet = True

    if hypot(pos[0] - pts[-1][0], pos[1] - pts[-1][1]) > 0.5:
        filled, pack_end = fill_straights_between(
            pos,
            pts[-1],
            kit,
            length_for_stem=length_for_stem,
        )
        pieces.extend(filled)
        pos = pack_end if filled else pts[-1]

    if kit.terminator and pieces:
        end_yaw: float = heading(pts[-2], pts[-1]) if len(pts) >= 2 else 0.0
        pieces.append(
            FittedPiece(
                stem=kit.terminator,
                x=pos[0],
                z=pos[1],
                yaw_deg=end_yaw,
            )
        )
    return pieces


def append_centerline_point(
    points: list[tuple[float, float]], x: float, z: float
) -> None:
    """append xz when it moves the path"""
    if points:
        last_x: float
        last_z: float
        last_x, last_z = points[-1]
        if hypot(x - last_x, z - last_z) < 1e-4:
            return
    points.append((x, z))


def fitted_centerline_xz(
    pieces: Sequence[FittedPiece],
    *,
    length_for_stem: Callable[[str], float],
    corner_for_stem: Callable[[str], CornerPiece | None],
    snaps_for_stem: Callable[[str], PieceSnapPoints | None] | None = None,
    step_m: float = 2.0,
) -> list[tuple[float, float]]:
    """densified centerline through fitted piece geometry (entry→exit)

    corners follow the cw mesh under placement yaw (ccw uses the reverse yaw-spin).
    when snaps exist, piece xz is the mesh origin so mid-arc locals are shifted.
    """
    spacing: float = max(0.5, step_m)
    out: list[tuple[float, float]] = []
    piece: FittedPiece
    for piece in pieces:
        corner: CornerPiece | None = corner_for_stem(piece.stem)
        if corner is not None:
            arc_len: float = float(corner.arc_length_m)
            steps: int = max(2, int(ceil(arc_len / spacing)))
            dx: float = 0.0
            dz: float = 0.0
            if snaps_for_stem is not None:
                snaps: PieceSnapPoints | None = snaps_for_stem(piece.stem)
                if snaps is not None:
                    clockwise: bool = abs(piece.roll_deg) < 90.0
                    dx, dz = corner_snap_origin_delta_local(
                        corner, snaps, clockwise=clockwise
                    )
            seg: list[tuple[float, float]] = []
            sample: int
            for sample in range(steps + 1):
                t: float = float(sample) / float(steps)
                lx: float
                lz: float
                lx, lz = corner_local_centerline_xz(corner, t)
                seg.append(
                    apply_yaw_xz(
                        piece.x,
                        piece.z,
                        piece.yaw_deg,
                        lx - dx,
                        lz - dz,
                    )
                )
            # ccw y-spin runs the cw mesh reverse; keep travel continuity
            if out and hypot(
                seg[0][0] - out[-1][0], seg[0][1] - out[-1][1]
            ) > hypot(
                seg[-1][0] - out[-1][0], seg[-1][1] - out[-1][1]
            ):
                seg.reverse()
            point: tuple[float, float]
            for point in seg:
                append_centerline_point(out, point[0], point[1])
            continue
        length: float = max(float(length_for_stem(piece.stem)), 0.5)
        ux: float
        uz: float
        ux, uz = heading_unit(piece.yaw_deg)
        half: float = length * 0.5
        steps = max(1, int(ceil(length / spacing)))
        sample = 0
        while sample <= steps:
            s: float = -half + length * (float(sample) / float(steps))
            append_centerline_point(
                out, piece.x + ux * s, piece.z + uz * s
            )
            sample += 1
    return out
