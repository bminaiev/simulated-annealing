use std::{
    fs::{self, File},
    io::Read,
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

use plotly::{
    common::{ColorScale, ColorScalePalette, Marker, Mode, Title},
    layout::{Axis, AxisType},
    HeatMap, Layout, Plot, Scatter, Scatter3D, Trace,
};
use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::random::Random;

mod random;

#[derive(Clone)]
struct Point {
    x: f64,
    y: f64,
}

struct SaParams {
    start_temp: f64,
    end_temp: f64,
}

struct MovingAverage {
    mult: f64,
    value: f64,
    max_sum: f64,
}

impl MovingAverage {
    pub fn new() -> Self {
        Self {
            mult: 0.9995,
            value: 0.0,
            max_sum: 0.0,
        }
    }

    pub fn add(&mut self, x: bool) {
        self.value *= self.mult;
        self.value += (if x { 1.0 } else { 0.0 }) * (1.0 - self.mult);
        self.max_sum *= self.mult;
        self.max_sum += 1.0 - self.mult;
    }

    pub fn get_value(&self) -> f64 {
        if self.max_sum == 0.0 {
            return 0.0;
        }
        return self.value / self.max_sum;
    }
}

impl SaParams {
    pub fn new() -> Self {
        Self {
            start_temp: 10.0,
            end_temp: 0.001,
        }
    }

    pub fn best() -> Self {
        Self {
            start_temp: 0.2,
            end_temp: 5e-3,
        }
    }

    pub fn gen_random(
        start_temp_range: std::ops::Range<f64>,
        end_temp_range: std::ops::Range<f64>,
    ) -> Self {
        let start_temp = start_temp_range.start
            * (start_temp_range.end / start_temp_range.start).powf(fastrand::f64());
        let end_temp = end_temp_range.start
            * (end_temp_range.end / end_temp_range.start).powf(fastrand::f64());
        if start_temp <= end_temp {
            return Self::gen_random(start_temp_range, end_temp_range);
        }
        Self {
            start_temp,
            end_temp,
        }
    }

    pub fn from(start_temp: f64, end_temp: f64) -> Self {
        Self {
            start_temp,
            end_temp,
        }
    }
}

// Parsing inputs in Rust is such a pain.
fn read_tsp100() -> Vec<Point> {
    let mut input = File::open("inputs/tsp100.txt").unwrap();
    let mut content = String::new();
    input.read_to_string(&mut content).unwrap();
    let tokens: Vec<_> = content.split_ascii_whitespace().collect();
    let n: usize = tokens[0].parse().unwrap();
    (0..n)
        .map(|id| Point {
            x: tokens[id * 2 + 1].parse().unwrap(),
            y: tokens[id * 2 + 2].parse().unwrap(),
        })
        .collect()
}

fn calc_dists(pts: &[Point]) -> Vec<Vec<f64>> {
    let n = pts.len();
    let mut res = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let dx = pts[i].x - pts[j].x;
            let dy = pts[i].y - pts[j].y;
            res[i][j] = (dx * dx + dy * dy).sqrt()
        }
    }
    res
}

fn calc_score(dists: &[Vec<f64>], perm: &[usize]) -> f64 {
    let n = dists.len();
    assert_eq!(perm.len(), n);
    let mut res = dists[perm[0]][perm[n - 1]];
    for w in perm.windows(2) {
        res += dists[w[0]][w[1]];
    }
    res
}

fn pick_random_interval_to_reverse(n: usize) -> (usize, usize) {
    let x = fastrand::usize(0..n);
    let y = fastrand::usize(0..n);
    if x < y {
        (x, y + 1)
    } else {
        (y, x + 1)
    }
}

const N: usize = 100;
fn pick_my_random_interval_to_reverse(rnd: &mut Random) -> (usize, usize) {
    let x = rnd.next_in_range(0, N);
    let y = rnd.next_in_range(0, N);
    if x < y {
        (x, y + 1)
    } else {
        (y, x + 1)
    }
}

fn line_and_scatter_plot() {
    let trace1 = Scatter::new(vec![1.0, 2.0, 3.0, 4.0], vec![10, 15, 13, 17])
        .name("trace1")
        .mode(Mode::Markers);

    let mut plot = Plot::new();
    plot.add_trace(trace1);
    fs::write("b.html", plot.to_html()).unwrap();
    // plot.show();
}

fn debug_plotly() -> bool {
    line_and_scatter_plot();
    true
}

fn save_html(traces: Vec<Box<dyn Trace>>, name: &str, y_axis_override: Option<&str>) {
    let y_axis_name = y_axis_override.unwrap_or("Score");
    let mut plot = Plot::new();
    plot.add_traces(traces);
    {
        let x_axis = Axis::new().title(Title::new("Time (s)"));
        let y_axis = Axis::new().title(Title::new(y_axis_name));
        plot.set_layout(
            Layout::new()
                .x_axis(x_axis)
                .y_axis(y_axis)
                .show_legend(false),
        );
    }
    let contents = if SAVE_PART_HTML {
        plot.to_inline_html(None)
    } else {
        plot.to_html()
    };
    fs::write(format!("htmls/{name}"), contents).unwrap();
}

struct SaResult {
    score_pts: Vec<Point>,
    best_pts: Vec<Point>,
    good_changes_pts: Vec<Point>,
    bad_changes_pts: Vec<Point>,
    final_score: f64,
    final_best_score: f64,
}

fn trace_from_points(pts: &[Point], name: Option<&str>) -> Box<dyn Trace> {
    let mut res = Scatter::new(
        pts.iter().map(|p| p.x).collect(),
        pts.iter().map(|p| p.y).collect(),
    )
    .mode(Mode::Lines);
    if let Some(name) = name {
        res = res.name(name.to_owned());
    }
    res
}

fn last_part(pts: &[Point], from: f64) -> Vec<Point> {
    pts.iter().filter(|p| p.x > from).cloned().collect()
}

impl SaResult {
    fn gen_trace(&self, name: Option<&str>) -> Box<dyn Trace> {
        trace_from_points(&self.score_pts, name)
    }

    fn save_html(&self, name: &str) {
        let trace = self.gen_trace(None);
        save_html(vec![trace], name, None);
    }
}

const MAX_SEC: f64 = 1.0;

fn run_sa(pts: &[Point], points_to_save: usize, sa_params: &SaParams, max_sec: f64) -> SaResult {
    let n = pts.len();
    let dists = calc_dists(&pts);
    let mut perm: Vec<_> = (0..n).collect();
    fastrand::shuffle(&mut perm);
    let mut prev_score = calc_score(&dists, &perm);
    let mut best_score = prev_score;

    let start = Instant::now();
    let mut score_pts = vec![];
    let mut best_pts = vec![];
    let mut good_changes_pts = vec![];
    let mut bad_changes_pts = vec![];

    let mut good_changes = MovingAverage::new();
    let mut bad_changes = MovingAverage::new();
    loop {
        let elapsed_s = start.elapsed().as_secs_f64();
        if elapsed_s > max_sec {
            break;
        }
        let elapsed_frac = elapsed_s / max_sec;

        let next_time_for_pts = (score_pts.len() as f64) / (1 + points_to_save) as f64;
        if elapsed_frac > next_time_for_pts {
            score_pts.push(Point {
                x: elapsed_s,
                y: prev_score,
            });
            best_pts.push(Point {
                x: elapsed_s,
                y: best_score,
            });
            good_changes_pts.push(Point {
                x: elapsed_s,
                y: good_changes.get_value(),
            });
            bad_changes_pts.push(Point {
                x: elapsed_s,
                y: bad_changes.get_value(),
            });
        }

        let temp =
            sa_params.start_temp * (sa_params.end_temp / sa_params.start_temp).powf(elapsed_frac);
        let (fr, to) = pick_random_interval_to_reverse(n);
        perm[fr..to].reverse();
        let new_score = calc_score(&dists, &perm);

        good_changes.add(new_score < prev_score);

        if new_score < prev_score || fastrand::f64() < ((prev_score - new_score) / temp).exp() {
            // Using new state!
            prev_score = new_score;
            if prev_score < best_score {
                best_score = prev_score;
            }
            bad_changes.add(new_score >= prev_score);
        } else {
            // Rollback
            perm[fr..to].reverse();
            bad_changes.add(false);
        }
    }
    let score = calc_score(&dists, &perm);
    eprintln!("Score: {score}, best: {best_score}");
    SaResult {
        score_pts,
        best_pts,
        good_changes_pts,
        bad_changes_pts,
        final_score: score,
        final_best_score: best_score,
    }
}

fn run_sa_optimized(pts: &[Point], sa_params: &SaParams, max_sec: f64) -> SaResult {
    let n = pts.len();
    assert_eq!(N, n);
    let dists_vec = calc_dists(&pts);
    let mut dists = [[0.0; N]; N];
    for i in 0..n {
        for j in 0..n {
            dists[i][j] = dists_vec[i][j];
        }
    }
    let mut perm: Vec<_> = (0..n).collect();
    fastrand::shuffle(&mut perm);
    let mut prev_score = calc_score(&dists_vec, &perm);
    let mut best_score = prev_score;

    let mut rnd = Random::new(787788);
    let start = Instant::now();

    let mut good_changes = MovingAverage::new();
    let mut bad_changes = MovingAverage::new();
    let mut checked_changes = 0i64;
    let mut elapsed_s = 0.0;
    let mut temp = 0.0;
    for iter in 0.. {
        if iter & 127 == 0 {
            elapsed_s = start.elapsed().as_secs_f64();

            let elapsed_frac = elapsed_s / max_sec;
            temp = sa_params.start_temp
                * (sa_params.end_temp / sa_params.start_temp).powf(elapsed_frac);
        }
        if elapsed_s > max_sec {
            break;
        }

        let (fr, to) = pick_my_random_interval_to_reverse(&mut rnd);
        if to - fr + 1 >= n {
            // this change does nothing.
            continue;
        }

        let v1 = if fr == 0 { perm[n - 1] } else { perm[fr - 1] };
        let v2 = perm[fr];
        let v3 = perm[to - 1];
        let v4 = if to == n { perm[0] } else { perm[to] };
        // we replace edges (v1, v2) and (v3, v4) with (v1, v3) and (v2, v4)
        let score_delta = dists[v1][v3] + dists[v2][v4] - dists[v1][v2] - dists[v3][v4];

        let new_score = prev_score + score_delta;
        good_changes.add(new_score < prev_score);

        if new_score < prev_score || rnd.gen_double() < ((prev_score - new_score) / temp).exp() {
            // Using new state!
            perm[fr..to].reverse();
            prev_score = new_score;
            if prev_score < best_score {
                best_score = prev_score;
            }
            bad_changes.add(new_score >= prev_score);
        } else {
            bad_changes.add(false);
        }
        checked_changes += 1;
    }
    let score = calc_score(&dists_vec, &perm);
    // eprintln!(
    //     "Average transitions checked: {}/s. Total checked: {checked_changes}.",
    //     (checked_changes as f64 / start.elapsed().as_secs_f64()) as i64
    // );
    // eprintln!("Score: {score}, best: {best_score}");
    SaResult {
        score_pts: vec![],
        best_pts: vec![],
        good_changes_pts: vec![],
        bad_changes_pts: vec![],
        final_score: score,
        final_best_score: best_score,
    }
}

fn first_version(pts: &[Point]) {
    let sa_res = run_sa(&pts, 300, &SaParams::new(), MAX_SEC);
    sa_res.save_html("first_version.html");
}

fn simple_with_sa_params(pts: &[Point], sa_params: &SaParams, name: &str) {
    let mut traces = vec![];
    for _ in 0..SEVERAL_TIMES {
        let sa_res = run_sa(pts, 300, sa_params, MAX_SEC);
        traces.push(trace_from_points(&sa_res.best_pts, None));
    }
    save_html(traces, name, None);
}

const SEVERAL_TIMES: usize = 5;

fn several_times(pts: &[Point]) {
    let mut traces = vec![];
    let mut traces08 = vec![];
    for _ in 0..SEVERAL_TIMES {
        let res = run_sa(&pts, 100, &SaParams::new(), MAX_SEC);
        traces.push(res.gen_trace(None));
        let last_part: Vec<Point> = last_part(&res.score_pts, 0.8);
        traces08.push(trace_from_points(&last_part, None));
    }
    save_html(traces, "several_times.html", None);
    save_html(traces08, "several_times_suf.html", None);
}

fn save_best(pts: &[Point]) {
    let sa_res = run_sa(&pts, 300, &SaParams::new(), MAX_SEC);
    {
        let best_trace = trace_from_points(&sa_res.best_pts, Some("best"));
        let cur_trace = trace_from_points(&sa_res.score_pts, Some("current"));
        let traces = vec![cur_trace, best_trace];
        save_html(traces, "cur_and_best.html", None);
    }
    {
        const PART: f64 = 0.8;
        let best_trace = trace_from_points(&last_part(&sa_res.best_pts, PART), Some("best"));
        let cur_trace = trace_from_points(&last_part(&sa_res.score_pts, PART), Some("current"));
        let traces = vec![cur_trace, best_trace];
        save_html(traces, "cur_and_best_suf.html", None);
    }
}

fn choosing_params(pts: &[Point], max_end_temp: f64, min_start_temp: f64, name: &str) {
    let mut plot = Plot::new();

    let mut x = vec![];
    let mut y = vec![];
    let mut z = vec![];

    let results: Vec<_> = (0..1000)
        .into_par_iter()
        .map(|_| {
            let sa_params = SaParams::gen_random(min_start_temp..100.0, 0.0001..max_end_temp);
            let sa_res = run_sa(&pts, 1, &sa_params, MAX_SEC);
            (
                sa_params.start_temp,
                sa_params.end_temp,
                sa_res.final_best_score,
            )
        })
        .collect();
    for (x1, y1, z1) in results.into_iter() {
        x.push(x1);
        y.push(y1);
        z.push(z1);
    }

    let trace = Scatter3D::new(x, y, z)
        .mode(Mode::Markers)
        .marker(Marker::new().size(3));

    plot.add_trace(trace);
    {
        let x_axis = Axis::new()
            .title(Title::new("Start Temp"))
            .type_(AxisType::Log);
        let y_axis = Axis::new()
            .title(Title::new("End Temp"))
            .type_(AxisType::Log);
        let z_axis = Axis::new().title(Title::new("Score"));
        plot.set_layout(
            Layout::new()
                .x_axis(x_axis)
                .y_axis(y_axis)
                .z_axis(z_axis)
                .show_legend(false),
        );
    }

    let contents = if SAVE_PART_HTML {
        plot.to_inline_html(None)
    } else {
        plot.to_html()
    };
    let contents = replace_3d_axis(contents);
    fs::write(format!("htmls/{name}.html"), contents).unwrap();
}

fn replace_3d_axis(s: String) -> String {
    let pattern = "\"layout\": {";
    let add = r#"
      "margin" : {
        "l" : 0,
        "r" : 0,
        "b" : 0,
        "t" : 0,
      },
      "height" : 700,
      "scene" : {
    "#;
    if let Some(pos) = s.find(pattern) {
        let mut res = s[0..pos].to_owned() + pattern + add;
        let mut brackets = 1i32;
        let mut added_additional_bracket = false;
        for c in s[pos + pattern.len()..].chars() {
            if c == '{' {
                brackets += 1;
            } else if c == '}' {
                brackets -= 1;
                if !added_additional_bracket && brackets == 0 {
                    added_additional_bracket = true;
                    res.push('}');
                }
            }
            res.push(c);
        }
        return res;
    }
    s
}

const OPTIMAL_RES: f64 = 7.83176;

fn optimial_answers(pts: &[Point], name: &str) {
    let mut x = vec![];
    let mut y = vec![];
    let mut z = vec![];

    let start_range = 0.1..100.0f64;
    let end_range = 0.0001..0.01f64;

    const MX: usize = 10;
    const PER_BUCKET: usize = 5000;
    let mut to_test = vec![];
    for i in 0..MX {
        for j in 0..MX {
            for _k in 0..PER_BUCKET {
                to_test.push((i, j));
            }
        }
    }
    fastrand::shuffle(&mut to_test);

    let mut ok_cnt = vec![vec![0; MX]; MX];

    let get_start_end = |xi: usize, yi: usize| -> (f64, f64) {
        let start_frac = ((xi as f64) + 0.5) / (MX as f64);
        let end_frac = ((yi as f64) + 0.5) / (MX as f64);

        let start = start_range.start * (start_range.end / start_range.start).powf(start_frac);
        let end = end_range.start * (end_range.end / end_range.start).powf(end_frac);

        (start, end)
    };

    let done = AtomicUsize::new(0);
    let start_timer = Instant::now();
    let results: Vec<_> = to_test
        .par_iter()
        .filter_map(|&(xi, yi)| {
            let (start, end) = get_start_end(xi, yi);
            let sa_params = SaParams::from(start, end);
            let res = run_sa(pts, 1, &sa_params, MAX_SEC);

            let done = done.fetch_add(1, Ordering::SeqCst);
            eprintln!(
                "Done: {done}/{}, elapsed: {:?}",
                to_test.len(),
                start_timer.elapsed()
            );

            if res.final_best_score > OPTIMAL_RES {
                return None;
            }
            return Some((xi, yi));
        })
        .collect();

    for (xi, yi) in results.into_iter() {
        ok_cnt[xi][yi] += 1;
    }

    for xi in 0..MX {
        for yi in 0..MX {
            let cur_z = (ok_cnt[xi][yi] as f64) / (PER_BUCKET as f64);

            let (start, end) = get_start_end(xi, yi);

            x.push(format!("{:.2}", start));
            y.push(format!("{:+.2e}", end));
            z.push(cur_z);
        }
    }

    let heatmap = HeatMap::new(x, y, z);

    let mut plot = Plot::new();
    plot.add_trace(heatmap);
    {
        let x_axis = Axis::new()
            .title(Title::new("Start Temp"))
            .type_(AxisType::Category);
        let y_axis = Axis::new()
            .title(Title::new("End Temp"))
            .type_(AxisType::Category);
        plot.set_layout(
            Layout::new()
                .x_axis(x_axis)
                .y_axis(y_axis)
                .show_legend(false),
        );
    }

    let contents = if SAVE_PART_HTML {
        plot.to_inline_html(None)
    } else {
        plot.to_html()
    };
    fs::write(format!("htmls/{name}.html"), contents).unwrap();
}

fn optimial_time_interval(pts: &[Point]) {
    let intervals_s = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0];
    // TODO: increase and rerun
    const MAX_TIME: f64 = 10000.0;
    let mut to_test = vec![];
    let mut cnt = vec![0; intervals_s.len()];
    for i in 0..intervals_s.len() {
        let times = (MAX_TIME / intervals_s[i]).round() as usize;
        cnt[i] = times;
        for _ in 0..times {
            to_test.push(i);
        }
    }
    fastrand::shuffle(&mut to_test);
    let done = AtomicUsize::new(0);
    let results: Vec<_> = to_test
        .par_iter()
        .filter_map(|&i| {
            let res = run_sa(pts, 1, &SaParams::best(), intervals_s[i]);

            let done = done.fetch_add(1, Ordering::SeqCst);
            eprintln!("Done: {done}/{}", to_test.len());

            if res.final_best_score > OPTIMAL_RES {
                return None;
            }
            return Some(i);
        })
        .collect();

    let mut ok = vec![0; intervals_s.len()];
    for &i in results.iter() {
        ok[i] += 1;
    }
    for i in 0..intervals_s.len() {
        let prob = (ok[i] as f64) / (cnt[i] as f64);
        let expected_time = intervals_s[i] / prob;
        println!(
            "| {}s | {:.1}% | {:.1}s",
            intervals_s[i],
            prob * 100.0,
            expected_time
        );
    }
}

fn better_start_params(pts: &[Point]) {
    simple_with_sa_params(&pts, &SaParams::best(), "better_start_params.html");
}

fn acceptance_prob(pts: &[Point], sa_params: &SaParams, name: &str) {
    let mut traces = vec![];
    let sa_res = run_sa(pts, 300, sa_params, MAX_SEC);
    traces.push(trace_from_points(
        &sa_res.good_changes_pts[1..],
        Some("good changes"),
    ));
    traces.push(trace_from_points(
        &sa_res.bad_changes_pts[1..],
        Some("bad accepted"),
    ));
    save_html(traces, &format!("{name}.html"), Some("Part of changes"));
}

const BEST_OPTIMIZED_TIME: f64 = 0.02;

fn check_optimized(pts: &[Point]) {
    let mut good = 0;
    for it in 0..1 {
        let res = run_sa_optimized(pts, &SaParams::best(), BEST_OPTIMIZED_TIME);
        if res.final_best_score < OPTIMAL_RES {
            good += 1;
            eprintln!("FOUND GOOD! {}/{}", good, it + 1);
        }
    }
}
fn part_of_optimals_optimized(pts: &[Point]) {
    const MAX_TIME: f64 = 1000.0;

    let total_tests = (MAX_TIME / BEST_OPTIMIZED_TIME) as i64;

    let done = AtomicUsize::new(0);
    let results: Vec<_> = (0..total_tests)
        .into_par_iter()
        .filter_map(|_| {
            let res = run_sa_optimized(pts, &SaParams::best(), BEST_OPTIMIZED_TIME);

            let done = done.fetch_add(1, Ordering::SeqCst);
            if done % 100 == 0 {
                eprintln!("Done: {done}/{total_tests}");
            }

            if res.final_best_score > OPTIMAL_RES {
                return None;
            }
            return Some(());
        })
        .collect();

    let prob = (results.len() as f64) / (total_tests as f64);
    let expected_time = BEST_OPTIMIZED_TIME / prob;
    println!(
        "| {}s | {:.1}% | {:.1}s",
        BEST_OPTIMIZED_TIME,
        prob * 100.0,
        expected_time
    );
}

const SAVE_PART_HTML: bool = false;
fn main() {
    // fastrand::seed(787788);
    let pts = read_tsp100();
    // first_version(&pts);

    // save_best(&pts);
    // choosing_params(&pts, 0.01, 0.0, &"choosing_params2");
    // choosing_params(&pts, 0.01, 0.1, &"choosing_params3");
    // optimial_answers(&pts, "optimal_answers_new2");
    // better_start_params(&pts);
    // optimial_time_interval(&pts);
    // acceptance_prob(&pts, &SaParams::best(), "acceptance_good");
    // acceptance_prob(&pts, &SaParams::new(), "acceptance_start");
    // check_optimized(&pts);
    part_of_optimals_optimized(&pts);
}
