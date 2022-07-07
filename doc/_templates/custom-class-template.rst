{{ name | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

    {% block attributes %}
    {% if attributes %}
    .. rubric:: {{ _('Attributes') }}

    .. autosummary::
        :toctree:
    {% for item in attributes %}
    {% if item.0 != item.upper().0 and item not in inherited_members %}
        {{ name }}.{{ item }}
    {% endif %}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block methods %}
    {% if methods %}
    .. rubric:: {{ _('Methods') }}

    .. autosummary::
        :toctree:
    {% for item in methods %}
    {% if item != "__init__" and item not in inherited_members %}
        {{ name }}.{{ item }}
    {% endif %}
    {%- endfor %}
    {% endif %}
    {% endblock %}
