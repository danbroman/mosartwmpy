import numba as nb


@nb.jit(
    "void("
        "int64, float64[:], float64[:], float64[:], float64[:],"
        "float64[:], float64[:], float64[:], float64"
    ")",
    nopython=True,
    nogil=True,
    cache=True,
)
def update_subnetwork_state(
    i,  # Index of the subnetwork element to update
    subnetwork_length,
    subnetwork_width,
    subnetwork_storage,
    subnetwork_cross_section_area,
    subnetwork_depth,
    subnetwork_wetness_perimeter,
    subnetwork_hydraulic_radii,
    tiny_value,
):
    """
    Updates the physical properties of the subnetwork river channels based on current state.

    Parameters:
    i (int64): Index of the subnetwork.
    subnetwork_length (float64[:]): Length of the subnetwork river channels.
    subnetwork_width (float64[:]): Width of the subnetwork river channels.
    subnetwork_storage (float64[:]): Storage of the subnetwork river channels.
    subnetwork_cross_section_area (float64[:]): Cross-sectional area of the subnetwork river channels.
    subnetwork_depth (float64[:]): Depth of the subnetwork river channels.
    subnetwork_wetness_perimeter (float64[:]): Wetness perimeter of the subnetwork river channels.
    subnetwork_hydraulic_radii (float64[:]): Hydraulic radii of the subnetwork river channels.
    tiny_value (float64): A small value used to avoid division by zero.
    """

    has_water = (subnetwork_length[i] > 0.0) and (subnetwork_storage[i] > 0.0)

    if has_water:
        subnetwork_cross_section_area[i] = subnetwork_storage[i] / subnetwork_length[i]
    else:
        subnetwork_cross_section_area[i] = 0.0

    if has_water and (subnetwork_cross_section_area[i] > tiny_value):
        subnetwork_depth[i] = subnetwork_cross_section_area[i] / subnetwork_width[i]
    else:
        subnetwork_depth[i] = 0.0

    if has_water and (subnetwork_depth[i] > tiny_value):
        subnetwork_wetness_perimeter[i] = subnetwork_width[i] + 2.0 * subnetwork_depth[i]
    else:
        subnetwork_wetness_perimeter[i] = 0.0

    if has_water and (subnetwork_wetness_perimeter[i] > tiny_value):
        subnetwork_hydraulic_radii[i] = subnetwork_cross_section_area[i] / subnetwork_wetness_perimeter[i]
    else:
        subnetwork_hydraulic_radii[i] = 0.0
