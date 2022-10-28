Handling deprecations
=====================

We attempt to adhere to semantic versioning as best we can. This means that as little,
ideally no, functionality should break between minor releases. Deprecation warnings are
raised whenever possible and feasible for functions/methods/properties/arguments, so
that users get a heads-up one (minor) release before something is removed or changes,
with a possible alternative to be used.

The decorator should be placed right above the object signature to be deprecated::

    @deprecate(since=0.8, removal=0.9, alternative="bar")
    def foo(self, n):
        return n + 1

    @property
    @deprecate(since=0.9, removal=0.10, alternative="another", object_type="property")
    def this_property(self):
        return 2