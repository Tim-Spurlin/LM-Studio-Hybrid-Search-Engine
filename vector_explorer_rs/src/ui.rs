// ui.rs — egui control panels, inspector, legend

use egui::*;
use crate::point_cloud::ColorMode;
use crate::search::{SearchState, FilterColumn};
use crate::data::Dataset;

pub struct UiState {
    pub color_mode:     ColorMode,
    pub point_size:     f32,
    pub search:         SearchState,
    pub selected_point: Option<usize>,
    pub show_legend:    bool,
    pub needs_rebuild:  bool,
}

impl UiState {
    pub fn new() -> Self {
        Self {
            color_mode:     ColorMode::ByTopic,
            point_size:     3.0,
            search:         SearchState::new(),
            selected_point: None,
            show_legend:    true,
            needs_rebuild:  false,
        }
    }
}

pub fn draw_left_panel(
    ctx:      &Context,
    state:    &mut UiState,
    dataset:  &Dataset,
    fps:      f32,
    n_points: u32,
) {
    SidePanel::left("controls")
        .exact_width(280.0)
        .resizable(false)
        .show(ctx, |ui| {

        ui.visuals_mut().window_fill = Color32::from_rgb(15, 15, 30);
        ui.visuals_mut().panel_fill  = Color32::from_rgb(15, 15, 30);
        ui.visuals_mut().override_text_color = Some(Color32::from_rgb(220, 220, 220));

        ui.heading(
            RichText::new("⬡ Vector Explorer")
                .color(Color32::from_rgb(126, 184, 247))
                .size(16.0)
        );
        ui.separator();

        // ── Stats ──
        ui.label(
            RichText::new(format!("Points: {}", n_points))
                .color(Color32::from_rgb(160, 196, 255))
                .size(11.0)
        );
        ui.label(
            RichText::new(format!("FPS: {:.0}", fps))
                .color(if fps > 55.0 {
                    Color32::from_rgb(100, 220, 100)
                } else {
                    Color32::from_rgb(220, 100, 100)
                })
                .size(11.0)
        );
        ui.separator();

        // ── Color Mode ──
        ui.label(RichText::new("Color By").color(Color32::from_rgb(126, 184, 247)));
        let prev_mode = state.color_mode.clone();
        ComboBox::from_id_salt("color_mode")
            .selected_text(format!("{:?}", state.color_mode))
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut state.color_mode, ColorMode::ByTopic,       "Topic");
                ui.selectable_value(&mut state.color_mode, ColorMode::ByStructure,   "Structure");
                ui.selectable_value(&mut state.color_mode, ColorMode::ByDomain,      "Domain");
                ui.selectable_value(&mut state.color_mode, ColorMode::ByVerified,    "Verified");
                ui.selectable_value(&mut state.color_mode, ColorMode::ByClusterSize, "Cluster Size");
                ui.selectable_value(&mut state.color_mode, ColorMode::Uniform,       "Uniform");
            });
        if state.color_mode != prev_mode {
            state.needs_rebuild = true;
        }
        ui.separator();

        // ── Point Size ──
        ui.label(RichText::new("Point Size").color(Color32::from_rgb(126, 184, 247)));
        ui.add(Slider::new(&mut state.point_size, 1.0..=12.0).step_by(0.5));
        ui.separator();

        // ── Search ──
        ui.label(RichText::new("Search").color(Color32::from_rgb(126, 184, 247)));
        let prev_query = state.search.query.clone();
        ui.text_edit_singleline(&mut state.search.query);

        if state.search.query != prev_query {
            state.search.run_search(dataset);
            state.needs_rebuild = true;
        }

        if state.search.match_count > 0 {
            ui.label(
                RichText::new(format!("{} matches highlighted in gold", state.search.match_count))
                    .color(Color32::from_rgb(255, 215, 0))
                    .size(10.0)
            );
        }
        ui.separator();

        // ── Filter ──
        ui.label(RichText::new("Filter").color(Color32::from_rgb(126, 184, 247)));

        let prev_col = state.search.filter_column.clone();
        ComboBox::from_id_salt("filter_col")
            .selected_text(format!("{:?}", state.search.filter_column))
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut state.search.filter_column, FilterColumn::None,       "None");
                ui.selectable_value(&mut state.search.filter_column, FilterColumn::Topic,      "Topic");
                ui.selectable_value(&mut state.search.filter_column, FilterColumn::Structure,  "Structure");
                ui.selectable_value(&mut state.search.filter_column, FilterColumn::Domain,     "Domain");
                ui.selectable_value(&mut state.search.filter_column, FilterColumn::SourceFile, "Source File");
                ui.selectable_value(&mut state.search.filter_column, FilterColumn::Verified,   "Verified");
            });

        let prev_val = state.search.filter_value.clone();
        ui.text_edit_singleline(&mut state.search.filter_value);

        if state.search.filter_column != prev_col || state.search.filter_value != prev_val {
            state.needs_rebuild = true;
        }

        ui.separator();

        // ── Controls Help ──
        ui.label(
            RichText::new(
                "Left-drag: rotate\nRight-drag/scroll: zoom\n\
                 Middle-drag: pan\nClick point: inspect\nR: reset camera"
            )
            .color(Color32::from_rgb(120, 120, 140))
            .size(10.0)
        );
    });
}

pub fn draw_right_panel(
    ctx:            &Context,
    state:          &UiState,
    dataset:        &Dataset,
    on_reset_cam:   &mut bool,
    _on_screenshot: &mut bool,
) {
    SidePanel::right("inspector")
        .exact_width(320.0)
        .resizable(false)
        .show(ctx, |ui| {

        ui.visuals_mut().panel_fill = Color32::from_rgb(10, 10, 22);

        ui.heading(
            RichText::new("Inspector")
                .color(Color32::from_rgb(126, 184, 247))
                .size(14.0)
        );
        ui.separator();

        if let Some(idx) = state.selected_point {
            if let Some(point) = dataset.points.get(idx) {
                ui.label(
                    RichText::new(&point.topic)
                        .color(Color32::from_rgb(255, 215, 0))
                        .size(13.0)
                        .strong()
                );
                ui.add_space(4.0);

                Grid::new("point_meta")
                    .num_columns(2)
                    .striped(true)
                    .show(ui, |ui| {
                        let kv = |ui: &mut Ui, k: &str, v: &str| {
                            ui.label(RichText::new(k).color(Color32::from_rgb(126, 184, 247)).size(10.0));
                            ui.label(RichText::new(v).color(Color32::from_rgb(200, 200, 200)).size(10.0));
                            ui.end_row();
                        };
                        kv(ui, "structure",    &point.structure);
                        kv(ui, "cluster_id",   &point.cluster_id.to_string());
                        kv(ui, "cluster_size", &point.cluster_size.to_string());
                        kv(ui, "verified",     if point.verified { "Yes" } else { "No" });
                        kv(ui, "domain",       &point.domain);
                        kv(ui, "source",       &point.source_file);
                    });

                ui.separator();
                ui.label(
                    RichText::new("Chunk Text")
                        .color(Color32::from_rgb(126, 184, 247))
                        .size(11.0)
                );
                ScrollArea::vertical()
                    .max_height(300.0)
                    .show(ui, |ui| {
                        ui.label(
                            RichText::new(&point.text_preview)
                                .color(Color32::from_rgb(180, 230, 180))
                                .size(10.0)
                                .family(FontFamily::Monospace)
                        );
                    });
            }
        } else {
            ui.label(
                RichText::new("Click any point\nto inspect its content")
                    .color(Color32::from_rgb(80, 80, 100))
                    .size(13.0)
            );
        }

        ui.separator();

        if ui.button("Reset Camera  [R]").clicked() {
            *on_reset_cam = true;
        }
    });
}
