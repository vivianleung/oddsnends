=============
Odds and Ends
=============


.. image:: https://img.shields.io/pypi/v/oddsnends.svg
        :target: https://pypi.python.org/pypi/oddsnends

.. image:: https://img.shields.io/travis/vivianleung/oddsnends.svg
        :target: https://travis-ci.com/vivianleung/oddsnends

.. image:: https://readthedocs.org/projects/oddsnends/badge/?version=latest
        :target: https://oddsnends.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status



Miscellaneous data science methods


* Free software: MIT license
* Documentation: https://oddsnends.readthedocs.io.


Features
========

Classes
-------
* ``AttrDict``

Types
-----
* ``IntervalType``
* ``LoggingLevels``
* ``NoneType``
* ``Numeric``
* ``SeriesType``
* ``TwoTupleInts``

Methods
-------

General

* ``default``
* ``defaults``
* ``is_null``
* ``msg``
* ``now``
* ``parse_literal_eval``


For collections

* ``agg``
* ``dict2list``
* ``drop_duplicates``
* ``dropna``
* ``pops``
* ``simplify``
* ``strictcollection``

For intervals

* ``calc_intervals``
* ``intervals2locs``
* ``setops_ranges``

Math

* ``rounding``
* ``ceiling``
* ``floor``

pandas tools

* ``assign``
* ``check_if_exists``
* ``dedup_alias``
* ``pipe_concat``
* ``get_level_uniques``
* ``pivot_indexed_table``
* ``reorder_cols``
* ``sort_levels``
* ``swap_index``


.. Credits
.. =======

.. This package was created with Cookiecutter_ and the ``audreyr/cookiecutter-pypackage``_ project template.

.. .. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. .. _``audreyr/cookiecutter-pypackage``: https://github.com/audreyr/cookiecutter-pypackage
