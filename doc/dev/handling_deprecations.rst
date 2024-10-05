Handling deprecations
=====================

We attempt to adhere to semantic versioning as best we can.
This means that as little, ideally no, functionality should break between minor
releases.
Deprecation warnings are raised whenever possible and feasible for
functions/methods/properties/arguments, so that users get a heads-up one (minor) release
before something is removed or changes, with a possible alternative to be used.

The decorator should be placed right above the object signature to be deprecated::

    @deprecate(since=0.8, removal=0.9, alternative="bar")
    def foo(self, n: int) -> int:
        return n + 1

    @property
    @deprecate(
        since=0.9, removal=0.10, alternative="another", alternative_is_function=False
    )
    def this_property(self) -> int:
        return 2

Parameters can be deprecated as well::

    @deprecate_argument(name="n", since=0.8, removal=0.9, alternative="m")
    def foo(self, n: int | None = None, m: int: int | None = None) -> int:
        if m is None:
            m = n
        return m + 1

This will raise a warning if ``n`` is passed.