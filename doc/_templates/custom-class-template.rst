{{ name | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

    {# Members inherited from kikuchipy base classes are listed, members inherited from
       other packages are not. See Sphinx configuration for details. #}

    {% block attributes %}
    {% set kp_inherited = kikuchipy_inherited_members.get(fullname, []) %}
    {% if attributes %}
    .. rubric:: {{ _('Attributes') }}

    .. autosummary::
        :toctree:
        :template: custom-attribute-template.rst
        {% for item in attributes %}
        {% if item.0 != item.upper().0 and (item not in inherited_members or item in kp_inherited) %}
        {{ name }}.{{ item }}
        {% endif %}
        {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block methods %}
    {% set kp_inherited = kikuchipy_inherited_members.get(fullname, []) %}
    {% if methods %}
    .. rubric:: {{ _('Methods') }}

    .. autosummary::
        :toctree:
        :template: custom-method-template.rst
        {% for item in methods %}
        {% if item != "__init__" and (item not in inherited_members or item in kp_inherited) %}
        {{ name }}.{{ item }}
        {% endif %}
        {%- endfor %}
    {% endif %}
    {% endblock %}

.. minigallery:: {{ module }}.{{ objname }}
    :add-heading: Examples using ``{{ objname }}``
