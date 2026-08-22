

def test_an_explicit_base_is_not_overwritten_by_a_disabled_spec():
    """Asking for 16 facets must give 16 facets.

    Disabled specs fill in whatever the base does not say, but the base has to
    win where it does -- otherwise `--facets 16` silently builds the 8-facet
    default and the caller never finds out.
    """
    from gemopt.geometry import CutParams, facet_count
    from gemopt.params import SearchSpace

    space = SearchSpace(base=CutParams(n_base=16, girdle_multiple=3))
    params = space.to_params(space.start())
    assert params.n_base == 16
    assert params.girdle_multiple == 3
    # +1 for the table.  The culet row ships disabled and defaults to a point,
    # so it contributes no facet unless the client asks for a flat one.
    assert facet_count(params) == 3 * 16 + 2 * 3 * 16 + 1

    default = SearchSpace()
    assert default.to_params(default.start()).n_base == 8


def test_locking_the_table_takes_it_out_of_the_search():
    """The client wants to dial this ratio in himself.

    Locking has to do two things: pin the value, and stop the optimiser
    spending a dimension on it -- an unused dimension is not free, differential
    evolution still explores it.
    """
    from gemopt.geometry import CutParams
    from gemopt.params import SearchSpace

    free_before = len(SearchSpace().free())
    space = SearchSpace().lock("table", 0.58)

    assert len(space.free()) == free_before - 1
    assert "table" not in [s.name for s in space.free()]
    assert space.to_params(space.start()).table == 0.58


def test_a_flat_girdle_drops_the_second_thickness_from_the_search():
    """One thickness means one parameter; searching for a second is dead
    weight that costs generations the angles would otherwise get."""
    from gemopt.geometry import CutParams
    from gemopt.params import SearchSpace

    flat = SearchSpace(base=CutParams(flat_girdle=True))
    scalloped = SearchSpace(base=CutParams(flat_girdle=False))

    names = [s.name for s in flat.free()]
    assert "girdle_thick_half" not in names
    assert "girdle_thick_main" in names
    assert len(scalloped.free()) == len(flat.free()) + 1
