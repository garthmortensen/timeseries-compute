API Documentation
=================

This section provides example usage of the main methods and classes in the Generalized Timeseries project.

Data Generator
==============

PriceSeriesGenerator
--------------------

Example usage of the `PriceSeriesGenerator` class:

.. code-block:: python

    from generalized_timeseries.data_generator import PriceSeriesGenerator

    start_date = "2023-01-01"
    end_date = "2023-01-10"
    anchor_prices = {"GM": 51.1, "LM": 2.2}
    generator = PriceSeriesGenerator(start_date=start_date, end_date=end_date)

    price_dict, price_df = generator.generate_prices(anchor_prices=anchor_prices)

    print(price_df.head())

Data Processor
==============

MissingDataHandler
------------------

Example usage of the `MissingDataHandler` class:

.. code-block:: python

    from generalized_timeseries.data_processor import MissingDataHandler

    data = {
        "A": [1, 2, None, 4, 5],
        "B": [None, 2, 3, None, 5],
        "C": [1, None, None, 4, 5]
    }
    df = pd.DataFrame(data)

    handler = MissingDataHandler(strategy="forward_fill")
    processed_df = handler.handle(df)

    print(processed_df)

MissingDataHandlerFactory
-------------------------

Example usage of the `MissingDataHandlerFactory` class:

.. code-block:: python

    from generalized_timeseries.data_processor import MissingDataHandlerFactory

    handler = MissingDataHandlerFactory.create_handler("drop")
    processed_df = handler.handle(df)

    print(processed_df)

Stats Model
===========

ModelFactory
------------

Example usage of the `ModelFactory` class:

.. code-block:: python

    from generalized_timeseries.stats_model import ModelFactory

    model = ModelFactory.create_model("arima", order=(1, 1, 1))
    model.fit(data)

    forecast = model.forecast(steps=10)
    print(forecast)

Tests
=====

Example usage of the test modules:

.. code-block:: python

    import pytest

    # Run all tests
    pytest.main()

    # Run specific test
    pytest.main(["tests/test_data_generator.py::test_generate_prices"])
