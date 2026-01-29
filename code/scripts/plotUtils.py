
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def build_events_df(start_time_sec, end_time_sec, transition_data,
                    ports_linetypes, ports_colors,
                    # start_poke_in_time_col_name="Start_Poke_in_time",
                    start_poke_in_time_col_name="P1_IN_Ephys_TS",
                    start_port_col_name="Start_Port"):
    mask = np.logical_and(
        start_time_sec<=transition_data[start_poke_in_time_col_name],
        transition_data[start_poke_in_time_col_name]<end_time_sec
    )
    subset_transition_data = transition_data[mask]
    event_time = subset_transition_data[start_poke_in_time_col_name]
    ports_names = subset_transition_data[start_port_col_name]
    event_line_type = [ports_linetypes[port_name]
                       for port_name in ports_names]
    event_color = [ports_colors[port_name]
                   for port_name in ports_names]
    answer = pd.DataFrame(dict(event_time=event_time,
                               event_line_type=event_line_type,
                               event_color=event_color))
    return answer

def add_events_vlines(fig, events_df):
    n_events = events_df.shape[0]
    for i in range(n_events):
        fig.add_vline(x=events_df.iloc[i]["event_time"],
                      line_dash=events_df.iloc[i]["event_line_type"],
                      line_color=events_df.iloc[i]["event_color"])

def get_times_non_epoched(times, epochs_times):
    # times \in (n_trials, n_time_points_per_trial, 1)
    # epochs_times \in n_trials
    # return \in n_trials * n_time_points_per_trial
    times_non_epoched = epochs_times[0] + times[0, :, 0]
    for r in range(1, len(epochs_times)):
        times_non_epoched = np.append(times_non_epoched,
                                      epochs_times[r] + times[r, :, 0])
    return times_non_epoched


def get_latents_non_epoched(latents):
    # latents \in (n_trials, n_time_points_per_trial, n_latents)
    # return \in (n_trials * n_time_points_per_trial, n_latents)
    n_trials = latents.shape[0]
    latents_non_epoched = latents[0, :, :]
    for r in range(1, n_trials):
        latents_non_epoched = np.vstack((latents_non_epoched, latents[r, :, :]))
    return latents_non_epoched

def getPlotNonEpochedLatents(times, latents_means, latents_stds,
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

