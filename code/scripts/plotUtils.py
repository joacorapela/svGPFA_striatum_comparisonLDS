
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import svGPFA.utils.miscUtils


def build_events_df(start_time_sec, end_time_sec, transition_data,
                    in_ports_linetypes, out_ports_linetypes, ports_colors,
                    poke_in_time_col_name="P1_IN_Ephys_TS",
                    poke_out_time_col_name="P1_OUT_Ephys_TS",
                    start_port_col_name="Start_Port"):
    mask = np.logical_and(
        start_time_sec<=transition_data[poke_in_time_col_name],
        transition_data[poke_out_time_col_name]<end_time_sec
    )
    subset_transition_data = transition_data[mask]
    event_time = np.concatenate(
        (subset_transition_data[poke_in_time_col_name].to_list(),
         subset_transition_data[poke_out_time_col_name].to_list()),
    )
    ports_names = subset_transition_data[start_port_col_name]
    in_line_type = [in_ports_linetypes[port_name]
                    for port_name in ports_names]
    out_line_type = [out_ports_linetypes[port_name]
                    for port_name in ports_names]
    event_line_type = in_line_type + out_line_type
    event_color = [ports_colors[port_name]
                   for port_name in ports_names]
    event_color = event_color + event_color
    answer = pd.DataFrame(dict(event_time=event_time,
                               event_line_type=event_line_type,
                               event_color=event_color))
    return answer

def buildMarkedEventsInfo(events_df):
    """
    answer:
        marked_events_trial_ids[r]: float, trial_id of the rth trial
        marked_events_times[r]: list of float, times of events in rth trial
        marked_events_colors[r]: list of floats, colors of events in rth trial
        marked_events_markers[r]: list of floats, markers of events in rth trial

    """

    trials_ids = np.unique(events_df["event_trial_id"])
    n_trials = len(trials_ids)
    marked_events_trial_ids = [None for r in range(n_trials)]
    marked_events_times = [None for r in range(n_trials)]
    marked_events_colors = [None for r in range(n_trials)]
    marked_events_markers = [None for r in range(n_trials)]

    for r, trial_id in enumerate(trials_ids):
        trial_events_df = events_df[events_df["event_trial_id"]==trial_id]
        n_items_in_trial = len(trial_events_df)
        marked_events_trial_ids[r] = trial_id
        trial_times = [None for i in range(n_items_in_trial)]
        trial_colors = [None for i in range(n_items_in_trial)]
        trial_markers = [None for i in range(n_items_in_trial)]
        for i in range(n_items_in_trial):
            trial_times[i] = trial_events_df.iloc[i]["event_time"]
            trial_colors[i] = trial_events_df.iloc[i]["event_color"]
            trial_markers[i] = trial_events_df.iloc[i]["event_marker"]
        marked_events_times[r] = trial_times
        marked_events_colors[r] = trial_colors
        marked_events_markers[r] = trial_markers
    return marked_events_trial_ids, marked_events_times, marked_events_colors, marked_events_markers

def buildMarkedEventsInfoFromTransitions(
        transitions_data, trials_ids, port_numbers=np.array([1, 2, 3, 4, 5, 6, 7]),
        port_colors=np.array(["green", "red", "cyan", "yellow", "purple",
                              "blue", "magenta"]),
        stage_markers=["cross", "circle"],
        trial_id_colname="Trial_id",
        port_in_time_colname="P1_IN_Ephys_TS",
        port_out_time_colname="P1_OUT_Ephys_TS",
        start_port_colname="Start_Port"):
    n_trials = len(trials_ids)
    marked_events_times = [None for r in range(n_trials)]
    marked_events_colors = [None for r in range(n_trials)]
    marked_events_markers = [None for r in range(n_trials)]
    marked_events_labels = [None for r in range(n_trials)]
    for r, trial_id in enumerate(trials_ids):
        trial_transitions_data = \
            transitions_data[transitions_data[trial_id_colname]==trial_id]
        trial_marked_events_times = []
        trial_marked_events_colors = []
        trial_marked_events_markers = []
        trial_marked_events_labels = []
        for i in range(trial_transitions_data.shape[0]):
            trial_marked_events_times.append(
                trial_transitions_data.iloc[i][port_in_time_colname],
            )
            trial_marked_events_times.append(
                trial_transitions_data.iloc[i][port_out_time_colname],
            )

            port_number_index = np.where(
                port_numbers == trial_transitions_data.iloc[i][start_port_colname])[0].item()

            trial_marked_events_colors.append(port_colors[port_number_index])
            trial_marked_events_colors.append(port_colors[port_number_index])

            trial_marked_events_markers.append(stage_markers[0])
            trial_marked_events_markers.append(stage_markers[1])

            trial_marked_events_labels.append(
                f"enter port {port_numbers[port_number_index]}")
            trial_marked_events_labels.append(
                f"exit port {port_numbers[port_number_index]}")

        marked_events_times[r] = trial_marked_events_times
        marked_events_colors[r] = trial_marked_events_colors
        marked_events_markers[r] = trial_marked_events_markers
        marked_events_labels[r] = trial_marked_events_labels
    return marked_events_times, marked_events_colors, marked_events_markers, \
           marked_events_labels

def add_events_vlines(fig, events_df):
    n_events = events_df.shape[0]
    for i in range(n_events):
        fig.add_vline(x=events_df.iloc[i]["event_time"],
                      line_dash=events_df.iloc[i]["event_line_type"],
                      line_color=events_df.iloc[i]["event_color"])

def getPlotNonEpochedLatents(times, latents_means, latents_stds,
                             trials_ids_for_samples,
                             events_names=None,
                             marked_events_times=None,
                             marked_events_colors=None,
                             marked_events_markers=None,
                             marked_size=10,
                             latents_colors_patterns=None,
                             default_latents_color_pattern="rgba(128,128,128,{:f})",
                             cb_transparency=0.3, mean_transparency=1.0,
                             xlabel="Time (sec)", ylabel="Value"):
    # times \in (n_time_points,)
    # latents_means \in (n_time_points, n_latents)

    n_latents = latents_means.shape[-1]
    fig = go.Figure()
    latents_ci = 1.96 * latents_stds

    x = times
    for k in range(n_latents):
        y = latents_means[:, k]
        latent_ci = latents_ci[:, k]
        y_upper = y + latent_ci
        y_lower = y - latent_ci
        ymax = max(np.max(y+latent_ci), np.max(y+latent_ci))
        ymin = min(np.min(y-latent_ci), np.min(y-latent_ci))

        if latents_colors_patterns is not None:
            latent_color_pattern = latents_colors_patterns[k]
        else:
            latent_color_pattern = default_latents_color_pattern

        traceCB = go.Scatter(
            x=np.concatenate((x, x[::-1])),
            y=np.concatenate((y_upper, y_lower[::-1])),
            fill="toself",
            fillcolor=latent_color_pattern.format(cb_transparency),
            line=dict(color=latent_color_pattern.format(0.0)),
            showlegend=False,
            legendgroup="latent{:02d}".format(k)
        )
        latent_label = "{:02d}".format(k)
        traceMean = go.Scatter(
            x=x,
            y=y,
            hovertemplate=
            "<b>time</b>=%{x} sec"+
            "<br><b>value</b>=%{y}"+
            "<br><b>trial</b>=%{text}",
            text=trials_ids_for_samples,
            line=dict(color=latent_color_pattern.format(mean_transparency)),
            mode="lines",
            name="latent {:s}".format(latent_label),
            legendgroup="latent{:02d}".format(k)
        )
        fig.add_trace(traceCB)
        fig.add_trace(traceMean)

        # add markers to trials
        if events_names is not None and\
           marked_events_times is not None and \
           marked_events_colors is not None and \
           marked_events_markers is not None and \
           align_event_times is not None:
            n_marked_events = len(marked_events_times[r])
            marked_events_times_centered = marked_events_times[r]-align_event_times[r]
            for i in range(n_marked_events):
                if not math.isnan(marked_events_times_centered[i]):
                    marked_index = np.argmin(np.abs(
                        times[r, :, 0]-marked_events_times_centered[i]))

                    trace_marker = go.Scatter(
                        x=[times[r, marked_index, 0]],
                        y=[meanToPlot[marked_index]],
                        marker=dict(color=marked_events_colors[r][i],
                                    symbol=marked_events_markers[r][i],
                                    size=marked_size),
                        mode="markers",
                        text=[events_names[i]],
                        hovertemplate="x=%{x}<br>" + "y=%{y}<br>" + "event=%{text}",
                        legendgroup="trial{:02d}".format(trials_ids[r]),
                        showlegend=False)
                    fig.add_trace(trace_marker)

    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text=ylabel)
    return fig


def getLatentsColorPatterns(n_latents,
                            color_patterns=["rgba(0,0,139,{:f})",       # darkblue
                                            "rgba(0,0,255,{:f})",       # blue
                                            "rgba(173,216,230,{:f})",   # lightblue
                                            "rgba(139,0,0,{:f})",       # darkred
                                            "rgba(255,0,0,{:f})",       # red
                                            "rgba(2,48,32,{:f})",       # darkgreen
                                            "rgba(144,238,144,{:f})",   # lightgreen
                                            "rgba(255,255,0,{:f})",     # yellow
                                            "rgba(255,255,224,{:f})",   # lightyellow
                                            "rgba(255,0,225,{:f})",     # magenta
                                            "rgba(0,255,225,{:f})"],    # cyan
                           ):
    if n_latents > len(color_patterns):
        raise ValueError("Insufficient number of colors "
                         "in argument color_patterns")
    return color_patterns[:n_latents]

def getPlotOrthonormalizedLatentAcrossTrials(
        times, latentsMeans, latentsVars, C, trials_ids,
        latentToPlot=0,
        align_event_times=None,
        rewarded_trials_times=None,
        short_trials_cluster_responsibilities=None,
        events_names=None,
        marked_events_times=None,
        marked_events_colors=None,
        marked_events_markers=None,
        marked_size=10,
        trials_colors_patterns=None,
        default_trial_color_pattern="rgba(128,128,128,{:f})",
        cb_transparency=0.3, mean_transparency=1.0,
        trials_annotations=None, ylim=None,
        xlabel="Time (sec)", ylabel="Value",
        titlePattern="Orthonormalized latent {:d}"):
    # align_event_times[r] \in double
    # marked_events_times[r] \in list of size n_events_r
    # marked_events_colors[r] \in list of size n_events_r
    # marked_events_markers[r] \in list of size n_events_r
    n_trials = len(latentsMeans)
    tLatentsMeans, tLatentsVars = svGPFA.utils.miscUtils.orthogonalizeLatents(
        latents_means=latentsMeans, latents_vars=latentsVars, C=C)

    if ylim is None:
        latents_max = -np.Inf
        latents_min = np.Inf
    for r in range(n_trials):
        if ylim is None:
            tLatentsMeansr_min = tLatentsMeans[r].min()
            tLatentsMeansr_max = tLatentsMeans[r].max()
            if tLatentsMeansr_min < latents_min:
                latents_min = tLatentsMeansr_min
            if tLatentsMeansr_max > latents_max:
                latents_max = tLatentsMeansr_max
    if ylim is None:
        ylim = [latents_min, latents_max]
    fig = go.Figure()
    title = titlePattern.format(latentToPlot)

    hover_texts = [["Trial: {:02d}<br>Time: {:f}".format(trial_id, time)
                    for i, time in enumerate(times[r, :, 0])]
                   for r, trial_id in enumerate(trials_ids)]
    if trials_annotations is not None:
        for r in range(n_trials):
            n_times = times.shape[1]
            an_annotation = ""
            for trial_annotation_key in trials_annotations:
                an_annotation += "<br>{:s}: {}".format(
                    trial_annotation_key,
                    trials_annotations[trial_annotation_key][r])
            for i in range(n_times):
                hover_texts[r][i] = hover_texts[r][i] + an_annotation

    for r in range(n_trials):
        trial_times = times[r, :, 0]
        meanToPlot = tLatentsMeans[r, :, latentToPlot]
        stdToPlot = np.sqrt(tLatentsVars[r, :, latentToPlot])
        ciToPlot = 1.96*stdToPlot
        if trials_colors_patterns is not None:
            trial_color_pattern = trials_colors_patterns[r]
        else:
            trial_color_pattern = default_trial_color_pattern

        x = trial_times
        y = meanToPlot
        y_upper = y + ciToPlot
        y_lower = y - ciToPlot
        ymax = max(np.max(meanToPlot+ciToPlot), np.max(meanToPlot+ciToPlot))
        ymin = min(np.min(meanToPlot-ciToPlot), np.min(meanToPlot-ciToPlot))

        traceCB = go.Scatter(
            x=np.concatenate((x, x[::-1])),
            y=np.concatenate((y_upper, y_lower[::-1])),
            fill="toself",
            fillcolor=trial_color_pattern.format(cb_transparency),
            line=dict(color=trial_color_pattern.format(0.0)),
            showlegend=False,
            legendgroup="trial{:02d}".format(trials_ids[r])
        )

        if trials_ids is not None:
            trial_label = "{:02d}".format(trials_ids[r])
        else:
            trial_label = "{:02d}".format(r)

        if align_event_times is not None:
            if rewarded_trials_times is not None and \
               short_trials_cluster_responsibilities is not None:
                # check if align_event_times[r] in rewarded_trials_times[:, 0]
                res = np.where(align_event_times[r]==
                                 rewarded_trials_times["start_time"])[0]
                if len(res) > 0:
                    index = res[0].item()
                    name = "trial {:02d}, e_time {:.02f}, resp {:.02f}".format(
                        trials_ids[r], align_event_times[r],
                        short_trials_cluster_responsibilities[index])
                else:
                    name = "trial {:02d}, e_time {:.02f}".format(
                        trials_ids[r], align_event_times[r])

            else:
                name = "trial {:02d}, e_time {:.02f}".format(
                    trials_ids[r], align_event_times[r])
        else:
            name = "trial {:02d}".format(trials_ids[r])

        traceMean = go.Scatter(
            x=x,
            y=y,
            line=dict(color=trial_color_pattern.format(mean_transparency)),
            mode="lines",
            name=name,
            legendgroup="trial{:02d}".format(trials_ids[r]),
            showlegend=True,
            hoverinfo="text",
            text=hover_texts[r],
        )
        fig.add_trace(traceCB)
        fig.add_trace(traceMean)

        # add markers to trials
        if marked_events_times is not None and \
                marked_events_colors is not None and \
                marked_events_markers is not None and \
                align_event_times is not None:
            n_marked_events = len(marked_events_times[r])
            marked_events_times_centered = marked_events_times[r]-align_event_times[r]
            for i in range(n_marked_events):
                if not math.isnan(marked_events_times_centered[i]):
                    marked_index = np.argmin(np.abs(
                        times[r, :, 0]-marked_events_times_centered[i]))

                    trace_marker = go.Scatter(
                        x=[times[r, marked_index, 0]],
                        y=[meanToPlot[marked_index]],
                        marker=dict(color=marked_events_colors[r][i],
                                    symbol=marked_events_markers[r][i],
                                    size=marked_size),
                        text=[events_names[i]],
                        hovertemplate="x=%{x}<br>" + "y=%{y}<br>" + "event=%{text}",
                        mode="markers",
                        legendgroup="trial{:02d}".format(trials_ids[r]),
                        showlegend=False)
                    fig.add_trace(trace_marker)

    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text=ylabel, range=ylim)
    fig.update_layout(title_text=title)
    return fig

