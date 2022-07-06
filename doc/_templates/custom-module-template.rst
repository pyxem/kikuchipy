{{ name | escape | underline}}

.. automodule:: {{ fullname }}

    {% block attributes %}
        {% if attributes %}
            .. rubric:: Module Attributes
            .. autosummary::
                :toctree:
                {% for item in attributes %}
                    {% if item in members %}
                        {{ item }}
                    {% endif %}
                {%- endfor %}
        {% endif %}
    {% endblock %}

    {% block functions %}
        {% if functions %}
            .. rubric:: {{ _('Functions') }}
            .. autosummary::
                :toctree:
                {% for item in functions %}
                    {% if item in members %}
                        {{ item }}
                    {% endif %}
                {%- endfor %}
        {% endif %}
    {% endblock %}

    {% block classes %}
        {% if classes %}
            .. rubric:: {{ _('Classes') }}
            .. autosummary::
                :toctree:
                :template: custom-class-template.rst
                {% for item in classes %}
                    {% if item in members %}
                        {{ item }}
                    {% endif %}
                {%- endfor %}
        {% endif %}
    {% endblock %}

    {% block exceptions %}
        {% if exceptions %}
            .. rubric:: {{ _('Exceptions') }}
            .. autosummary::
                :toctree:
                {% for item in exceptions %}
                    {% if item in members %}
                        {{ item }}
                    {% endif %}
                {%- endfor %}
        {% endif %}
    {% endblock %}

{% block modules %}
    {% if modules %}
        .. rubric:: {{ _('Modules') }}
        .. autosummary::
            :toctree:
            :template: custom-module-template.rst
            :recursive:
            {% for item in modules %}
                {% if item in members %}
                    {{ item }}
                {% endif %}
            {%- endfor %}
    {% endif %}
{% endblock %}
