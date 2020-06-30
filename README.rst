Progetto di Bioinformatica 2019/2020
=========================================================================================
|travis| |sonar_quality| |sonar_maintainability| |codacy|
|code_climate_maintainability| |pip| |downloads|

Introduzione
----------------------------------------------
Since some software handling coverages sometimes
get slightly different results, here's three of them:

How do I install this package?
----------------------------------------------
As usual, just download it using pip:

.. code:: shell

    pip install progettobioinf

How do I run this project?
----------------------------------------------
Docker: build image and run container (docker_run.sh):
.. code:: shell

    docker build -t image-progettobioinf-app . && docker run --name container-progettobioinf image-progettobioinf-app

Tests Coverage
----------------------------------------------
Since some software handling coverages sometimes
get slightly different results, here's three of them:

|coveralls| |sonar_coverage| |code_climate_coverage|


.. |travis| image:: https://travis-ci.org/nicolasfacchinetti97/progettoBioInf.png
   :target: https://travis-ci.org/nicolasfacchinetti97/progettoBioInf
   :alt: Travis CI build

.. |sonar_quality| image:: https://sonarcloud.io/api/project_badges/measure?project=nicolasfacchinetti97_progettoBioInf&metric=alert_status
    :target: https://sonarcloud.io/dashboard/index/nicolasfacchinetti97_progettoBioInf
    :alt: SonarCloud Quality

.. |sonar_maintainability| image:: https://sonarcloud.io/api/project_badges/measure?project=nicolasfacchinetti97_progettoBioInf&metric=sqale_rating
    :target: https://sonarcloud.io/dashboard/index/nicolasfacchinetti97_progettoBioInf
    :alt: SonarCloud Maintainability

.. |sonar_coverage| image:: https://sonarcloud.io/api/project_badges/measure?project=nicolasfacchinetti97_progettoBioInf&metric=coverage
    :target: https://sonarcloud.io/dashboard/index/nicolasfacchinetti97_progettoBioInf
    :alt: SonarCloud Coverage

.. |coveralls| image:: https://coveralls.io/repos/github/nicolasfacchinetti97/progettoBioInf/badge.svg?branch=master
    :target: https://coveralls.io/github/nicolasfacchinetti97/progettoBioInf?branch=master
    :alt: Coveralls Coverage

.. |pip| image:: https://badge.fury.io/py/progettobioinf.svg
    :target: https://badge.fury.io/py/progettobioinf
    :alt: Pypi project

.. |downloads| image:: https://pepy.tech/badge/progettobioinf
    :target: https://pepy.tech/project/progettobioinf
    :alt: Pypi total project downloads

.. |codacy| image:: https://api.codacy.com/project/badge/Grade/280d48f738c34ac4a1cddec6f106480e
    :target: https://www.codacy.com/manual/nicolasfacchinetti97/progettoBioInf?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=nicolasfacchinetti97/progettoBioInf&amp;utm_campaign=Badge_Grade
    :alt: Codacy Maintainability

.. |code_climate_maintainability| image:: https://api.codeclimate.com/v1/badges/b70b9bb1eece3d914158/maintainability
    :target: https://codeclimate.com/github/nicolasfacchinetti97/progettoBioInf/maintainability
    :alt: Maintainability

.. |code_climate_coverage| image:: https://api.codeclimate.com/v1/badges/b70b9bb1eece3d914158/test_coverage
    :target: https://codeclimate.com/github/nicolasfacchinetti97/progettoBioInf/test_coverage
    :alt: Code Climate Coverage
