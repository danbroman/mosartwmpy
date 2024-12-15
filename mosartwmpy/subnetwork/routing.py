import numba as nb

from mosartwmpy.subnetwork.state import update_subnetwork_state
from mosartwmpy.utilities.timing import timing


# @timing
@nb.jit(
    "void("
        "int64, float64, int64, int64,"
        "int64[:], int64[:], float64[:], float64[:], float64[:], float64[:], float64[:],"
        "boolean[:], float64[:], float64[:], float64[:], float64[:], float64[:],"
        "float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],"
        "float64"
    ")",
    parallel=True,
    nopython=True,
    nogil=True,
    cache=True,
)
def subnetwork_routing(
    n,
    delta_t,
    routing_iterations,
    max_iterations_subnetwork,
    iterations_subnetwork,
    mosart_mask,
    subnetwork_slope,
    subnetwork_manning,
    subnetwork_length,
    subnetwork_width,
    hillslope_length,
    euler_mask,
    channel_lateral_flow_hillslope,
    subnetwork_flow_velocity,
    subnetwork_discharge,
    subnetwork_lateral_inflow,
    subnetwork_storage,
    subnetwork_storage_previous_timestep,
    subnetwork_delta_storage,
    subnetwork_depth,
    subnetwork_cross_section_area,
    subnetwork_wetness_perimeter,
    subnetwork_hydraulic_radii,
    tiny_value,
):
    """
    Tracks the storage and flow of water in the subnetwork river channels.

    Parameters:
    n (int64): Number of subnetwork segments.
    delta_t (float64): Time step for the simulation.
    routing_iterations (int64): Number of routing iterations.
    max_iterations_subnetwork (int64): Maximum number of iterations for the subnetwork.
    iterations_subnetwork (int64[:]): Array of iterations for each subnetwork segment.
    mosart_mask (int64[:]): Mask indicating active subnetwork segments.
    subnetwork_slope (float64[:]): Slope of each subnetwork segment.
    subnetwork_manning (float64[:]): Manning's roughness coefficient for each subnetwork segment.
    subnetwork_length (float64[:]): Length of each subnetwork segment.
    subnetwork_width (float64[:]): Width of each subnetwork segment.
    hillslope_length (float64[:]): Length of the hillslope for each subnetwork segment.
    euler_mask (boolean[:]): Mask indicating segments using Euler's method.
    channel_lateral_flow_hillslope (float64[:]): Lateral flow from the hillslope to the channel.
    subnetwork_flow_velocity (float64[:]): Flow velocity in each subnetwork segment.
    subnetwork_discharge (float64[:]): Discharge in each subnetwork segment.
    subnetwork_lateral_inflow (float64[:]): Lateral inflow into each subnetwork segment.
    subnetwork_storage (float64[:]): Storage in each subnetwork segment.
    subnetwork_storage_previous_timestep (float64[:]): Storage in each subnetwork segment from the previous timestep.
    subnetwork_delta_storage (float64[:]): Change in storage for each subnetwork segment.
    subnetwork_depth (float64[:]): Depth of water in each subnetwork segment.
    subnetwork_cross_section_area (float64[:]): Cross-sectional area of each subnetwork segment.
    subnetwork_wetness_perimeter (float64[:]): Wetness perimeter of each subnetwork segment.
    subnetwork_hydraulic_radii (float64[:]): Hydraulic radii of each subnetwork segment.
    tiny_value (float64): A small value to avoid division by zero.

    Returns:
    None
    """

    for i in nb.prange(n):

        local_delta_t = (delta_t / routing_iterations) / iterations_subnetwork[i]
        channel_lateral_flow_hillslope[i] = 0.0

        if not euler_mask[i] or not (mosart_mask[i] > 0):
            continue

        has_tributaries = subnetwork_length[i] > hillslope_length[i]

        # step through max iterations
        for _ in nb.prange(max_iterations_subnetwork):
            if not (iterations_subnetwork[i] > _):
                continue

            if has_tributaries:
                if subnetwork_hydraulic_radii[i] > 0.0:
                    subnetwork_flow_velocity[i] = (subnetwork_hydraulic_radii[i] ** (2.0/3.0)) * (subnetwork_slope[i] ** (1.0/2.0)) / subnetwork_manning[i]
                else:
                    subnetwork_flow_velocity[i] = 0.0
                subnetwork_discharge[i] = -subnetwork_flow_velocity[i] * subnetwork_cross_section_area[i]
            else:
                subnetwork_discharge[i] = -1.0 * subnetwork_lateral_inflow[i]

            discharge_condition = has_tributaries and ((subnetwork_storage[i] + (subnetwork_lateral_inflow[i] + subnetwork_discharge[i]) * local_delta_t) < tiny_value)

            if discharge_condition:
                subnetwork_discharge[i] = -(subnetwork_lateral_inflow[i] + subnetwork_storage[i] / local_delta_t)

            if discharge_condition and (subnetwork_cross_section_area[i] > 0.0):
                subnetwork_flow_velocity[i] = -subnetwork_discharge[i] / subnetwork_cross_section_area[i]

            subnetwork_delta_storage[i] = subnetwork_lateral_inflow[i] + subnetwork_discharge[i]

            # update storage
            subnetwork_storage_previous_timestep[i] = subnetwork_storage[i]
            subnetwork_storage[i] = subnetwork_storage[i] + subnetwork_delta_storage[i] * local_delta_t

            # update subnetwork state
            update_subnetwork_state(
                i,
                subnetwork_length,
                subnetwork_width,
                subnetwork_storage,
                subnetwork_cross_section_area,
                subnetwork_depth,
                subnetwork_wetness_perimeter,
                subnetwork_hydraulic_radii,
                tiny_value,
            )

            channel_lateral_flow_hillslope[i] = channel_lateral_flow_hillslope[i] - subnetwork_discharge[i]

        # average lateral flow over substeps
        channel_lateral_flow_hillslope[i] = channel_lateral_flow_hillslope[i] / iterations_subnetwork[i]
