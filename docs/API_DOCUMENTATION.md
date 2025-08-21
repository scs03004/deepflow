# API Documentation

**Project**: deepflow
**Generated**: 2025-08-21
**Framework**: Unknown

## Overview

This document provides comprehensive API documentation for deepflow.

### Base URL
- **Development**: `http://localhost:8000`
- **Production**: `https://your-domain.com`

## Endpoints


### PARAMETRIZE method

**Handler**: `test_series_values` in `venv\Lib\site-packages\pandas\tests\copy_view\test_array.py`


**Parameters**:

- `using_copy_on_write`

- `method`



**Dependencies**:

- shares_memory

- parametrize

- raises

- array

- copy

- method

- Series

- get_array

- asarray

- assert_series_equal




---


### PARAMETRIZE method

**Handler**: `test_dataframe_values` in `venv\Lib\site-packages\pandas\tests\copy_view\test_array.py`


**Parameters**:

- `using_copy_on_write`

- `using_array_manager`

- `method`



**Dependencies**:

- shares_memory

- parametrize

- raises

- assert_frame_equal

- array

- DataFrame

- copy

- method

- get_array

- asarray




---


### PARAMETRIZE order

**Handler**: `test_ravel_read_only` in `venv\Lib\site-packages\pandas\tests\copy_view\test_array.py`


**Parameters**:

- `using_copy_on_write`

- `order`



**Dependencies**:

- shares_memory

- parametrize

- ravel

- Series

- get_array

- assert_produces_warning




---


### PARAMETRIZE dtype

**Handler**: `test_astype_avoids_copy` in `venv\Lib\site-packages\pandas\tests\copy_view\test_astype.py`


**Parameters**:

- `using_copy_on_write`

- `dtype`

- `new_dtype`



**Dependencies**:

- shares_memory

- parametrize

- assert_frame_equal

- importorskip

- DataFrame

- copy

- get_array

- astype




---


### PARAMETRIZE dtype

**Handler**: `test_astype_different_target_dtype` in `venv\Lib\site-packages\pandas\tests\copy_view\test_astype.py`


**Parameters**:

- `using_copy_on_write`

- `dtype`



**Dependencies**:

- shares_memory

- parametrize

- assert_frame_equal

- importorskip

- DataFrame

- copy

- get_array

- _has_no_reference

- astype




---


### PARAMETRIZE dtype, new_dtype

**Handler**: `test_astype_string_and_object` in `venv\Lib\site-packages\pandas\tests\copy_view\test_astype.py`


**Parameters**:

- `using_copy_on_write`

- `dtype`

- `new_dtype`



**Dependencies**:

- shares_memory

- parametrize

- assert_frame_equal

- DataFrame

- copy

- get_array

- astype




---


### PARAMETRIZE dtype, new_dtype

**Handler**: `test_astype_string_and_object_update_original` in `venv\Lib\site-packages\pandas\tests\copy_view\test_astype.py`


**Parameters**:

- `using_copy_on_write`

- `dtype`

- `new_dtype`



**Dependencies**:

- shares_memory

- parametrize

- assert_frame_equal

- DataFrame

- copy

- get_array

- astype




---


### PARAMETRIZE func, args

**Handler**: `test_methods_iloc_getitem_item_cache` in `venv\Lib\site-packages\pandas\tests\copy_view\test_chained_assignment_deprecation.py`


**Parameters**:

- `func`

- `args`

- `using_copy_on_write`

- `warn_copy_on_write`



**Dependencies**:

- getattr

- parametrize

- DataFrame

- copy

- assert_cow_warning

- raises_chained_assignment_error




---


### PARAMETRIZE indexer

**Handler**: `test_series_setitem` in `venv\Lib\site-packages\pandas\tests\copy_view\test_chained_assignment_deprecation.py`


**Parameters**:

- `indexer`

- `using_copy_on_write`

- `warn_copy_on_write`



**Dependencies**:

- parametrize

- array

- DataFrame

- slice

- warns

- len




---


### FILTERWARNINGS ignore::pandas.errors.SettingWithCopyWarning

**Handler**: `test_frame_setitem` in `venv\Lib\site-packages\pandas\tests\copy_view\test_chained_assignment_deprecation.py`


**Parameters**:

- `indexer`

- `using_copy_on_write`



**Dependencies**:

- filterwarnings

- parametrize

- option_context

- array

- DataFrame

- raises_chained_assignment_error

- slice




---


### PARAMETRIZE dtype

**Handler**: `test_series_from_series` in `venv\Lib\site-packages\pandas\tests\copy_view\test_constructors.py`


**Parameters**:

- `dtype`

- `using_copy_on_write`

- `warn_copy_on_write`



**Dependencies**:

- shares_memory

- parametrize

- assert_cow_warning

- Series

- get_array

- has_reference




---


### PARAMETRIZE fastpath

**Handler**: `test_series_from_array` in `venv\Lib\site-packages\pandas\tests\copy_view\test_constructors.py`


**Parameters**:

- `using_copy_on_write`

- `idx`

- `dtype`

- `fastpath`

- `arr`



**Dependencies**:

- getattr

- parametrize

- shares_memory

- array

- copy

- Series

- get_array

- assert_produces_warning

- assert_series_equal

- RangeIndex




---


### PARAMETRIZE copy

**Handler**: `test_series_from_array_different_dtype` in `venv\Lib\site-packages\pandas\tests\copy_view\test_constructors.py`


**Parameters**:

- `using_copy_on_write`

- `copy`



**Dependencies**:

- shares_memory

- parametrize

- array

- Series

- get_array




---


### PARAMETRIZE idx

**Handler**: `test_series_from_index` in `venv\Lib\site-packages\pandas\tests\copy_view\test_constructors.py`


**Parameters**:

- `using_copy_on_write`

- `idx`



**Dependencies**:

- assert_index_equal

- parametrize

- shares_memory

- PeriodIndex

- Index

- Timedelta

- Period

- copy

- TimedeltaIndex

- Series

- get_array

- _has_no_reference

- Timestamp

- DatetimeIndex




---


### FILTERWARNINGS ignore:Setting a value on a view:FutureWarning

**Handler**: `test_series_from_block_manager` in `venv\Lib\site-packages\pandas\tests\copy_view\test_constructors.py`


**Parameters**:

- `using_copy_on_write`

- `idx`

- `dtype`

- `fastpath`



**Dependencies**:

- filterwarnings

- parametrize

- shares_memory

- copy

- Series

- get_array

- assert_produces_warning

- _has_no_reference

- assert_series_equal

- RangeIndex




---


### PARAMETRIZE use_mgr

**Handler**: `test_dataframe_constructor_mgr_or_df` in `venv\Lib\site-packages\pandas\tests\copy_view\test_constructors.py`


**Parameters**:

- `using_copy_on_write`

- `warn_copy_on_write`

- `columns`

- `use_mgr`



**Dependencies**:

- shares_memory

- parametrize

- assert_frame_equal

- DataFrame

- copy

- assert_cow_warning

- get_array

- assert_produces_warning




---


### PARAMETRIZE dtype

**Handler**: `test_dataframe_from_dict_of_series` in `venv\Lib\site-packages\pandas\tests\copy_view\test_constructors.py`


**Parameters**:

- `request`

- `using_copy_on_write`

- `warn_copy_on_write`

- `columns`

- `index`

- `dtype`



**Dependencies**:

- shares_memory

- parametrize

- assert_frame_equal

- DataFrame

- copy

- assert_cow_warning

- Series

- get_array

- assert_series_equal




---


### PARAMETRIZE dtype

**Handler**: `test_dataframe_from_dict_of_series_with_reindex` in `venv\Lib\site-packages\pandas\tests\copy_view\test_constructors.py`


**Parameters**:

- `dtype`



**Dependencies**:

- shares_memory

- parametrize

- DataFrame

- Series

- get_array




---


### PARAMETRIZE cons

**Handler**: `test_dataframe_from_series_or_index` in `venv\Lib\site-packages\pandas\tests\copy_view\test_constructors.py`


**Parameters**:

- `using_copy_on_write`

- `warn_copy_on_write`

- `data`

- `dtype`

- `cons`



**Dependencies**:

- shares_memory

- parametrize

- DataFrame

- assert_equal

- copy

- assert_cow_warning

- get_array

- _has_no_reference

- cons




---


### PARAMETRIZE cons

**Handler**: `test_dataframe_from_series_or_index_different_dtype` in `venv\Lib\site-packages\pandas\tests\copy_view\test_constructors.py`


**Parameters**:

- `using_copy_on_write`

- `cons`



**Dependencies**:

- shares_memory

- parametrize

- DataFrame

- get_array

- _has_no_reference

- cons




---


### PARAMETRIZE index

**Handler**: `test_dataframe_from_dict_of_series_with_dtype` in `venv\Lib\site-packages\pandas\tests\copy_view\test_constructors.py`


**Parameters**:

- `index`



**Dependencies**:

- shares_memory

- parametrize

- DataFrame

- Series

- get_array




---


### PARAMETRIZE copy

**Handler**: `test_frame_from_numpy_array` in `venv\Lib\site-packages\pandas\tests\copy_view\test_constructors.py`


**Parameters**:

- `using_copy_on_write`

- `copy`

- `using_array_manager`



**Dependencies**:

- shares_memory

- parametrize

- array

- DataFrame

- get_array




---


### PARAMETRIZE copy

**Handler**: `test_concat_copy_keyword` in `venv\Lib\site-packages\pandas\tests\copy_view\test_functions.py`


**Parameters**:

- `using_copy_on_write`

- `copy`



**Dependencies**:

- shares_memory

- parametrize

- DataFrame

- concat

- get_array




---


### PARAMETRIZE func

**Handler**: `test_merge_on_key` in `venv\Lib\site-packages\pandas\tests\copy_view\test_functions.py`


**Parameters**:

- `using_copy_on_write`

- `func`



**Dependencies**:

- merge

- shares_memory

- parametrize

- assert_frame_equal

- DataFrame

- copy

- Series

- get_array

- func




---


### PARAMETRIZE func, how

**Handler**: `test_merge_on_key_enlarging_one` in `venv\Lib\site-packages\pandas\tests\copy_view\test_functions.py`


**Parameters**:

- `using_copy_on_write`

- `func`

- `how`



**Dependencies**:

- merge

- shares_memory

- parametrize

- assert_frame_equal

- DataFrame

- copy

- Series

- get_array

- func

- _has_no_reference




---


### PARAMETRIZE copy

**Handler**: `test_merge_copy_keyword` in `venv\Lib\site-packages\pandas\tests\copy_view\test_functions.py`


**Parameters**:

- `using_copy_on_write`

- `copy`



**Dependencies**:

- merge

- shares_memory

- parametrize

- DataFrame

- get_array




---


### PARAMETRIZE dtype

**Handler**: `test_join_on_key` in `venv\Lib\site-packages\pandas\tests\copy_view\test_functions.py`


**Parameters**:

- `dtype`

- `using_copy_on_write`



**Dependencies**:

- shares_memory

- parametrize

- Index

- assert_frame_equal

- DataFrame

- copy

- get_array

- join




---


### PARAMETRIZE dtype

**Handler**: `test_subset_column_slice` in `venv\Lib\site-packages\pandas\tests\copy_view\test_indexing.py`


**Parameters**:

- `backend`

- `using_copy_on_write`

- `warn_copy_on_write`

- `using_array_manager`

- `dtype`



**Dependencies**:

- shares_memory

- parametrize

- assert_frame_equal

- option_context

- array

- DataFrame

- copy

- assert_cow_warning

- get_array

- _verify_integrity

- assert_produces_warning




---


### PARAMETRIZE dtype

**Handler**: `test_subset_loc_rows_columns` in `venv\Lib\site-packages\pandas\tests\copy_view\test_indexing.py`


**Parameters**:

- `backend`

- `dtype`

- `row_indexer`

- `column_indexer`

- `using_array_manager`

- `using_copy_on_write`

- `warn_copy_on_write`



**Dependencies**:

- parametrize

- assert_frame_equal

- array

- DataFrame

- copy

- assert_cow_warning

- range

- isinstance

- slice




---


### PARAMETRIZE dtype

**Handler**: `test_subset_iloc_rows_columns` in `venv\Lib\site-packages\pandas\tests\copy_view\test_indexing.py`


**Parameters**:

- `backend`

- `dtype`

- `row_indexer`

- `column_indexer`

- `using_array_manager`

- `using_copy_on_write`

- `warn_copy_on_write`



**Dependencies**:

- parametrize

- assert_frame_equal

- array

- DataFrame

- copy

- assert_cow_warning

- range

- isinstance

- slice




---


### PARAMETRIZE indexer

**Handler**: `test_subset_set_with_row_indexer` in `venv\Lib\site-packages\pandas\tests\copy_view\test_indexing.py`


**Parameters**:

- `backend`

- `indexer_si`

- `indexer`

- `using_copy_on_write`

- `warn_copy_on_write`



**Dependencies**:

- skip

- parametrize

- assert_frame_equal

- indexer_si

- array

- DataFrame

- option_context

- copy

- range

- assert_cow_warning

- isinstance

- assert_produces_warning

- slice




---


### PARAMETRIZE dtype

**Handler**: `test_subset_set_column_with_loc` in `venv\Lib\site-packages\pandas\tests\copy_view\test_indexing.py`


**Parameters**:

- `backend`

- `using_copy_on_write`

- `warn_copy_on_write`

- `using_array_manager`

- `dtype`



**Dependencies**:

- parametrize

- assert_frame_equal

- option_context

- array

- DataFrame

- copy

- range

- assert_cow_warning

- _verify_integrity

- assert_produces_warning




---


### PARAMETRIZE dtype

**Handler**: `test_subset_set_columns` in `venv\Lib\site-packages\pandas\tests\copy_view\test_indexing.py`


**Parameters**:

- `backend`

- `using_copy_on_write`

- `warn_copy_on_write`

- `dtype`



**Dependencies**:

- parametrize

- all

- assert_frame_equal

- option_context

- array

- DataFrame

- copy

- range

- _verify_integrity

- assert_produces_warning

- _has_no_reference

- astype




---


### PARAMETRIZE indexer

**Handler**: `test_subset_set_with_column_indexer` in `venv\Lib\site-packages\pandas\tests\copy_view\test_indexing.py`


**Parameters**:

- `backend`

- `indexer`

- `using_copy_on_write`

- `warn_copy_on_write`



**Dependencies**:

- parametrize

- assert_frame_equal

- option_context

- array

- DataFrame

- copy

- range

- assert_cow_warning

- _verify_integrity

- slice




---


### PARAMETRIZE method

**Handler**: `test_subset_chained_getitem` in `venv\Lib\site-packages\pandas\tests\copy_view\test_indexing.py`


**Parameters**:

- `request`

- `backend`

- `method`

- `dtype`

- `using_copy_on_write`

- `using_array_manager`

- `warn_copy_on_write`



**Dependencies**:

- endswith

- parametrize

- assert_frame_equal

- array

- DataFrame

- copy

- assert_cow_warning

- method




---


### PARAMETRIZE dtype

**Handler**: `test_subset_chained_getitem_column` in `venv\Lib\site-packages\pandas\tests\copy_view\test_indexing.py`


**Parameters**:

- `backend`

- `dtype`

- `using_copy_on_write`

- `warn_copy_on_write`



**Dependencies**:

- parametrize

- assert_frame_equal

- array

- DataFrame

- copy

- assert_cow_warning

- Series

- assert_series_equal

- _clear_item_cache




---


### PARAMETRIZE method

**Handler**: `test_subset_chained_getitem_series` in `venv\Lib\site-packages\pandas\tests\copy_view\test_indexing.py`


**Parameters**:

- `backend`

- `method`

- `using_copy_on_write`

- `warn_copy_on_write`



**Dependencies**:

- parametrize

- copy

- assert_cow_warning

- method

- Series

- assert_series_equal




---


### PARAMETRIZE method

**Handler**: `test_null_slice` in `venv\Lib\site-packages\pandas\tests\copy_view\test_indexing.py`


**Parameters**:

- `backend`

- `method`

- `using_copy_on_write`

- `warn_copy_on_write`



**Dependencies**:

- parametrize

- assert_frame_equal

- DataFrame

- copy

- assert_cow_warning

- method




---


### PARAMETRIZE method

**Handler**: `test_null_slice_series` in `venv\Lib\site-packages\pandas\tests\copy_view\test_indexing.py`


**Parameters**:

- `backend`

- `method`

- `using_copy_on_write`

- `warn_copy_on_write`



**Dependencies**:

- parametrize

- copy

- assert_cow_warning

- method

- Series

- assert_series_equal




---


### PARAMETRIZE indexer

**Handler**: `test_series_subset_set_with_indexer` in `venv\Lib\site-packages\pandas\tests\copy_view\test_indexing.py`


**Parameters**:

- `backend`

- `indexer_si`

- `indexer`

- `using_copy_on_write`

- `warn_copy_on_write`



**Dependencies**:

- parametrize

- slice

- indexer_si

- array

- copy

- assert_cow_warning

- Series

- isinstance

- assert_produces_warning

- assert_series_equal




---


### PARAMETRIZE method

**Handler**: `test_column_as_series_no_item_cache` in `venv\Lib\site-packages\pandas\tests\copy_view\test_indexing.py`


**Parameters**:

- `request`

- `backend`

- `method`

- `using_copy_on_write`

- `warn_copy_on_write`

- `using_array_manager`



**Dependencies**:

- parametrize

- assert_frame_equal

- option_context

- DataFrame

- copy

- assert_cow_warning

- method

- assert_produces_warning

- assert_series_equal




---


### PARAMETRIZE val

**Handler**: `test_set_value_copy_only_necessary_column` in `venv\Lib\site-packages\pandas\tests\copy_view\test_indexing.py`


**Parameters**:

- `using_copy_on_write`

- `warn_copy_on_write`

- `indexer_func`

- `indexer`

- `val`

- `col`



**Dependencies**:

- shares_memory

- parametrize

- assert_frame_equal

- DataFrame

- copy

- assert_cow_warning

- get_array

- assert_produces_warning

- slice

- indexer_func




---


### PARAMETRIZE dtype

**Handler**: `test_iset_splits_blocks_inplace` in `venv\Lib\site-packages\pandas\tests\copy_view\test_internals.py`


**Parameters**:

- `using_copy_on_write`

- `locs`

- `arr`

- `dtype`



**Dependencies**:

- shares_memory

- parametrize

- assert_frame_equal

- array

- DataFrame

- iset

- enumerate

- copy

- Series

- get_array

- astype




---


### PARAMETRIZE method

**Handler**: `test_interpolate_no_op` in `venv\Lib\site-packages\pandas\tests\copy_view\test_interp_fillna.py`


**Parameters**:

- `using_copy_on_write`

- `method`



**Dependencies**:

- shares_memory

- parametrize

- assert_frame_equal

- DataFrame

- copy

- get_array

- assert_produces_warning

- interpolate




---


### PARAMETRIZE func

**Handler**: `test_interp_fill_functions` in `venv\Lib\site-packages\pandas\tests\copy_view\test_interp_fillna.py`


**Parameters**:

- `using_copy_on_write`

- `func`



**Dependencies**:

- getattr

- parametrize

- shares_memory

- assert_frame_equal

- DataFrame

- copy

- get_array




---


### PARAMETRIZE func

**Handler**: `test_interpolate_triggers_copy` in `venv\Lib\site-packages\pandas\tests\copy_view\test_interp_fillna.py`


**Parameters**:

- `using_copy_on_write`

- `vals`

- `func`



**Dependencies**:

- getattr

- parametrize

- shares_memory

- DataFrame

- get_array

- _has_no_reference

- Timestamp




---


### PARAMETRIZE vals

**Handler**: `test_interpolate_inplace_no_reference_no_copy` in `venv\Lib\site-packages\pandas\tests\copy_view\test_interp_fillna.py`


**Parameters**:

- `using_copy_on_write`

- `vals`



**Dependencies**:

- shares_memory

- parametrize

- DataFrame

- get_array

- _has_no_reference

- Timestamp

- interpolate




---


### PARAMETRIZE vals

**Handler**: `test_interpolate_inplace_with_refs` in `venv\Lib\site-packages\pandas\tests\copy_view\test_interp_fillna.py`


**Parameters**:

- `using_copy_on_write`

- `vals`

- `warn_copy_on_write`



**Dependencies**:

- shares_memory

- parametrize

- assert_frame_equal

- DataFrame

- copy

- assert_cow_warning

- get_array

- _has_no_reference

- Timestamp

- interpolate




---


### PARAMETRIZE func

**Handler**: `test_interp_fill_functions_inplace` in `venv\Lib\site-packages\pandas\tests\copy_view\test_interp_fillna.py`


**Parameters**:

- `using_copy_on_write`

- `func`

- `warn_copy_on_write`

- `dtype`



**Dependencies**:

- getattr

- parametrize

- shares_memory

- assert_frame_equal

- DataFrame

- copy

- assert_cow_warning

- get_array

- _has_no_reference




---


### PARAMETRIZE downcast

**Handler**: `test_fillna_inplace` in `venv\Lib\site-packages\pandas\tests\copy_view\test_interp_fillna.py`


**Parameters**:

- `using_copy_on_write`

- `downcast`



**Dependencies**:

- shares_memory

- parametrize

- DataFrame

- fillna

- get_array

- assert_produces_warning

- _has_no_reference




---


### PARAMETRIZE func

**Handler**: `test_interpolate_chained_assignment` in `venv\Lib\site-packages\pandas\tests\copy_view\test_interp_fillna.py`


**Parameters**:

- `using_copy_on_write`

- `func`



**Dependencies**:

- getattr

- parametrize

- assert_frame_equal

- option_context

- DataFrame

- copy

- assert_produces_warning

- raises_chained_assignment_error




---


### PARAMETRIZE copy

**Handler**: `test_methods_copy_keyword` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `request`

- `method`

- `copy`

- `using_copy_on_write`

- `using_array_manager`



**Dependencies**:

- set_axis

- DataFrame

- reindex_like

- align

- infer_objects

- shares_memory

- to_timestamp

- assert_produces_warning

- set_flags

- truncate

- reindex

- swapaxes

- date_range

- get_array

- startswith

- tz_convert

- parametrize

- period_range

- tz_localize

- to_period

- method

- rename_axis

- rename

- astype




---


### PARAMETRIZE copy

**Handler**: `test_methods_series_copy_keyword` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `request`

- `method`

- `copy`

- `using_copy_on_write`



**Dependencies**:

- set_axis

- reindex_like

- align

- infer_objects

- shares_memory

- Series

- to_timestamp

- assert_produces_warning

- set_flags

- truncate

- reindex

- from_arrays

- swapaxes

- swaplevel

- date_range

- get_array

- tz_convert

- parametrize

- period_range

- tz_localize

- to_period

- method

- rename_axis

- rename

- astype




---


### PARAMETRIZE copy

**Handler**: `test_transpose_copy_keyword` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `copy`

- `using_array_manager`



**Dependencies**:

- shares_memory

- parametrize

- DataFrame

- transpose

- get_array




---


### PARAMETRIZE index

**Handler**: `test_reset_index_series_drop` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `index`



**Dependencies**:

- shares_memory

- parametrize

- Index

- copy

- Series

- get_array

- _has_no_reference

- reset_index

- assert_series_equal

- RangeIndex




---


### PARAMETRIZE index

**Handler**: `test_reindex_rows` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `index`

- `using_copy_on_write`



**Dependencies**:

- shares_memory

- parametrize

- assert_frame_equal

- DataFrame

- copy

- list

- get_array

- view

- index

- reindex




---


### PARAMETRIZE filter_kwargs

**Handler**: `test_filter` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `filter_kwargs`



**Dependencies**:

- shares_memory

- parametrize

- assert_frame_equal

- DataFrame

- copy

- get_array

- filter




---


### PARAMETRIZE func

**Handler**: `test_align_frame` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `func`



**Dependencies**:

- shares_memory

- parametrize

- assert_frame_equal

- DataFrame

- align

- copy

- get_array

- func

- slice




---


### PARAMETRIZE ax

**Handler**: `test_swapaxes_noop` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `ax`



**Dependencies**:

- shares_memory

- parametrize

- swapaxes

- assert_frame_equal

- DataFrame

- copy

- get_array

- assert_produces_warning




---


### PARAMETRIZE method, idx

**Handler**: `test_chained_methods` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `request`

- `method`

- `idx`

- `using_copy_on_write`

- `warn_copy_on_write`



**Dependencies**:

- parametrize

- assert_frame_equal

- DataFrame

- copy

- assert_cow_warning

- method

- rename

- select_dtypes

- reset_index




---


### PARAMETRIZE obj

**Handler**: `test_to_timestamp` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `obj`



**Dependencies**:

- shares_memory

- parametrize

- Index

- DataFrame

- Period

- assert_equal

- copy

- Series

- get_array

- to_timestamp




---


### PARAMETRIZE obj

**Handler**: `test_to_period` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `obj`



**Dependencies**:

- shares_memory

- parametrize

- Index

- DataFrame

- to_period

- assert_equal

- copy

- Series

- get_array

- Timestamp




---


### PARAMETRIZE axis, val

**Handler**: `test_dropna` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `axis`

- `val`



**Dependencies**:

- shares_memory

- parametrize

- assert_frame_equal

- DataFrame

- dropna

- copy

- get_array




---


### PARAMETRIZE val

**Handler**: `test_dropna_series` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `val`



**Dependencies**:

- shares_memory

- parametrize

- dropna

- copy

- Series

- assert_series_equal




---


### PARAMETRIZE method

**Handler**: `test_head_tail` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `method`

- `using_copy_on_write`

- `warn_copy_on_write`



**Dependencies**:

- shares_memory

- parametrize

- tail

- assert_frame_equal

- DataFrame

- copy

- assert_cow_warning

- method

- head

- get_array

- _verify_integrity




---


### PARAMETRIZE kwargs

**Handler**: `test_truncate` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `kwargs`



**Dependencies**:

- shares_memory

- parametrize

- assert_frame_equal

- DataFrame

- copy

- get_array

- _verify_integrity

- truncate




---


### PARAMETRIZE method

**Handler**: `test_assign_drop_duplicates` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `method`



**Dependencies**:

- getattr

- parametrize

- shares_memory

- assert_frame_equal

- DataFrame

- copy

- get_array

- _verify_integrity




---


### PARAMETRIZE obj

**Handler**: `test_take` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `obj`



**Dependencies**:

- shares_memory

- parametrize

- DataFrame

- assert_equal

- copy

- take

- Series




---


### PARAMETRIZE obj

**Handler**: `test_between_time` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `obj`



**Dependencies**:

- shares_memory

- parametrize

- between_time

- DataFrame

- assert_equal

- copy

- date_range

- Series




---


### PARAMETRIZE obj, kwargs

**Handler**: `test_sort_values` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `obj`

- `kwargs`



**Dependencies**:

- shares_memory

- parametrize

- DataFrame

- assert_equal

- copy

- Series

- get_array

- sort_values




---


### PARAMETRIZE obj, kwargs

**Handler**: `test_sort_values_inplace` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `obj`

- `kwargs`

- `warn_copy_on_write`



**Dependencies**:

- shares_memory

- parametrize

- DataFrame

- assert_equal

- copy

- assert_cow_warning

- Series

- get_array

- sort_values




---


### PARAMETRIZE decimals

**Handler**: `test_round` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `warn_copy_on_write`

- `decimals`



**Dependencies**:

- round

- shares_memory

- parametrize

- assert_frame_equal

- DataFrame

- copy

- get_array




---


### PARAMETRIZE obj

**Handler**: `test_swaplevel` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `obj`



**Dependencies**:

- shares_memory

- parametrize

- DataFrame

- assert_equal

- swaplevel

- copy

- Series

- from_tuples




---


### PARAMETRIZE kwargs

**Handler**: `test_rename_axis` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `kwargs`



**Dependencies**:

- shares_memory

- parametrize

- assert_frame_equal

- Index

- DataFrame

- copy

- rename_axis

- get_array




---


### PARAMETRIZE func, tz

**Handler**: `test_tz_convert_localize` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `func`

- `tz`



**Dependencies**:

- getattr

- parametrize

- shares_memory

- copy

- Series

- date_range

- assert_series_equal




---


### PARAMETRIZE dtype

**Handler**: `test_putmask` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `dtype`

- `warn_copy_on_write`



**Dependencies**:

- shares_memory

- parametrize

- assert_frame_equal

- DataFrame

- copy

- assert_cow_warning

- get_array




---


### PARAMETRIZE dtype

**Handler**: `test_putmask_no_reference` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `dtype`



**Dependencies**:

- DataFrame

- parametrize

- shares_memory

- get_array




---


### PARAMETRIZE dtype

**Handler**: `test_putmask_aligns_rhs_no_reference` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `dtype`



**Dependencies**:

- DataFrame

- parametrize

- shares_memory

- get_array




---


### PARAMETRIZE val, exp, warn

**Handler**: `test_putmask_dont_copy_some_blocks` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `val`

- `exp`

- `warn`

- `warn_copy_on_write`



**Dependencies**:

- shares_memory

- parametrize

- assert_frame_equal

- DataFrame

- copy

- list

- assert_cow_warning

- get_array

- assert_produces_warning

- _has_no_reference




---


### PARAMETRIZE dtype

**Handler**: `test_where_mask_noop` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `dtype`

- `func`



**Dependencies**:

- shares_memory

- parametrize

- where

- copy

- Series

- get_array

- mask

- func

- assert_series_equal




---


### PARAMETRIZE dtype

**Handler**: `test_where_mask` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `dtype`

- `func`



**Dependencies**:

- shares_memory

- parametrize

- where

- copy

- Series

- get_array

- mask

- func

- assert_series_equal




---


### PARAMETRIZE dtype, val

**Handler**: `test_where_mask_noop_on_single_column` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `dtype`

- `val`

- `func`



**Dependencies**:

- shares_memory

- parametrize

- assert_frame_equal

- DataFrame

- where

- copy

- get_array

- mask

- func




---


### PARAMETRIZE func

**Handler**: `test_chained_where_mask` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `func`



**Dependencies**:

- getattr

- parametrize

- assert_frame_equal

- option_context

- DataFrame

- copy

- assert_produces_warning

- raises_chained_assignment_error




---


### PARAMETRIZE dtype

**Handler**: `test_isetitem_series` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `dtype`



**Dependencies**:

- shares_memory

- parametrize

- isetitem

- assert_frame_equal

- array

- DataFrame

- copy

- Series

- get_array

- _has_no_reference

- assert_series_equal




---


### PARAMETRIZE key

**Handler**: `test_get` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `warn_copy_on_write`

- `key`



**Dependencies**:

- shares_memory

- parametrize

- assert_frame_equal

- option_context

- DataFrame

- copy

- get_array

- isinstance

- get

- assert_produces_warning




---


### PARAMETRIZE axis, key

**Handler**: `test_xs` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `warn_copy_on_write`

- `using_array_manager`

- `axis`

- `key`

- `dtype`



**Dependencies**:

- shares_memory

- parametrize

- assert_frame_equal

- option_context

- array

- DataFrame

- copy

- assert_cow_warning

- get_array

- assert_produces_warning

- xs

- _has_no_reference




---


### PARAMETRIZE axis

**Handler**: `test_xs_multiindex` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `warn_copy_on_write`

- `using_array_manager`

- `key`

- `level`

- `axis`



**Dependencies**:

- shares_memory

- parametrize

- from_product

- assert_frame_equal

- option_context

- DataFrame

- arange

- transpose

- reshape

- copy

- list

- get_array

- assert_produces_warning

- xs




---


### PARAMETRIZE copy

**Handler**: `test_transpose` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `copy`

- `using_array_manager`



**Dependencies**:

- shares_memory

- parametrize

- assert_frame_equal

- DataFrame

- transpose

- copy

- get_array




---


### PARAMETRIZE replace_kwargs

**Handler**: `test_replace` in `venv\Lib\site-packages\pandas\tests\copy_view\test_replace.py`


**Parameters**:

- `using_copy_on_write`

- `replace_kwargs`



**Dependencies**:

- replace

- shares_memory

- parametrize

- all

- assert_frame_equal

- DataFrame

- copy

- get_array




---


### PARAMETRIZE to_replace

**Handler**: `test_replace_inplace` in `venv\Lib\site-packages\pandas\tests\copy_view\test_replace.py`


**Parameters**:

- `using_copy_on_write`

- `to_replace`



**Dependencies**:

- replace

- shares_memory

- parametrize

- DataFrame

- get_array

- _has_no_reference




---


### PARAMETRIZE to_replace

**Handler**: `test_replace_inplace_reference` in `venv\Lib\site-packages\pandas\tests\copy_view\test_replace.py`


**Parameters**:

- `using_copy_on_write`

- `to_replace`

- `warn_copy_on_write`



**Dependencies**:

- replace

- shares_memory

- parametrize

- DataFrame

- assert_cow_warning

- get_array

- _has_no_reference




---


### PARAMETRIZE to_replace

**Handler**: `test_replace_inplace_reference_no_op` in `venv\Lib\site-packages\pandas\tests\copy_view\test_replace.py`


**Parameters**:

- `using_copy_on_write`

- `to_replace`



**Dependencies**:

- replace

- shares_memory

- parametrize

- DataFrame

- get_array

- _has_no_reference




---


### PARAMETRIZE to_replace

**Handler**: `test_replace_categorical_inplace_reference` in `venv\Lib\site-packages\pandas\tests\copy_view\test_replace.py`


**Parameters**:

- `using_copy_on_write`

- `val`

- `to_replace`



**Dependencies**:

- replace

- shares_memory

- parametrize

- assert_frame_equal

- Categorical

- DataFrame

- copy

- get_array

- assert_produces_warning

- _has_no_reference




---


### PARAMETRIZE val

**Handler**: `test_replace_categorical_inplace` in `venv\Lib\site-packages\pandas\tests\copy_view\test_replace.py`


**Parameters**:

- `using_copy_on_write`

- `val`



**Dependencies**:

- replace

- shares_memory

- parametrize

- assert_frame_equal

- Categorical

- DataFrame

- get_array

- assert_produces_warning

- _has_no_reference




---


### PARAMETRIZE val

**Handler**: `test_replace_categorical` in `venv\Lib\site-packages\pandas\tests\copy_view\test_replace.py`


**Parameters**:

- `using_copy_on_write`

- `val`



**Dependencies**:

- replace

- shares_memory

- parametrize

- assert_frame_equal

- Categorical

- DataFrame

- copy

- get_array

- assert_produces_warning

- _has_no_reference




---


### PARAMETRIZE method

**Handler**: `test_masking_inplace` in `venv\Lib\site-packages\pandas\tests\copy_view\test_replace.py`


**Parameters**:

- `using_copy_on_write`

- `method`

- `warn_copy_on_write`



**Dependencies**:

- getattr

- parametrize

- shares_memory

- assert_frame_equal

- DataFrame

- copy

- assert_cow_warning

- method

- get_array

- _has_no_reference




---


### PARAMETRIZE value

**Handler**: `test_replace_object_list_inplace` in `venv\Lib\site-packages\pandas\tests\copy_view\test_replace.py`


**Parameters**:

- `using_copy_on_write`

- `value`



**Dependencies**:

- replace

- shares_memory

- parametrize

- DataFrame

- get_array

- _has_no_reference




---


### PARAMETRIZE df

**Handler**: `test_empty_like` in `venv\Lib\site-packages\pandas\tests\frame\test_api.py`


**Parameters**:

- `df`



**Dependencies**:

- DataFrame

- parametrize




---


### PARAMETRIZE allows_duplicate_labels

**Handler**: `test_set_flags` in `venv\Lib\site-packages\pandas\tests\frame\test_api.py`


**Parameters**:

- `allows_duplicate_labels`

- `frame_or_series`

- `using_copy_on_write`

- `warn_copy_on_write`



**Dependencies**:

- parametrize

- DataFrame

- assert_cow_warning

- may_share_memory

- set_flags




---


### PARAMETRIZE name

**Handler**: `test_deprecations` in `venv\Lib\site-packages\pandas\tests\internals\test_api.py`


**Parameters**:

- `name`



**Dependencies**:

- getattr

- parametrize

- assert_produces_warning




---


### PARAMETRIZE key

**Handler**: `test_select_bad_cols` in `venv\Lib\site-packages\pandas\tests\resample\test_resample_api.py`


**Parameters**:

- `key`

- `test_frame`



**Dependencies**:

- raises

- parametrize

- resample




---


### PARAMETRIZE attr

**Handler**: `test_api_compat_before_use` in `venv\Lib\site-packages\pandas\tests\resample\test_resample_api.py`


**Parameters**:

- `attr`



**Dependencies**:

- getattr

- parametrize

- resample

- arange

- date_range

- Series

- len

- mean




---


### PARAMETRIZE on

**Handler**: `test_transform_frame` in `venv\Lib\site-packages\pandas\tests\resample\test_resample_api.py`


**Parameters**:

- `on`



**Dependencies**:

- parametrize

- resample

- assert_frame_equal

- datetime

- groupby

- DataFrame

- random

- default_rng

- transform

- list

- date_range

- reset_index

- Grouper




---


### PARAMETRIZE func

**Handler**: `test_apply_without_aggregation` in `venv\Lib\site-packages\pandas\tests\resample\test_resample_api.py`


**Parameters**:

- `func`

- `_test_series`



**Dependencies**:

- parametrize

- resample

- apply

- groupby

- func

- assert_series_equal

- Grouper




---


### PARAMETRIZE agg

**Handler**: `test_agg_both_mean_std_named_result` in `venv\Lib\site-packages\pandas\tests\resample\test_resample_api.py`


**Parameters**:

- `cases`

- `a_mean`

- `b_std`

- `agg`



**Dependencies**:

- parametrize

- assert_frame_equal

- aggregate

- concat

- assert_produces_warning

- NamedAgg




---


### PARAMETRIZE agg

**Handler**: `test_agg_both_mean_sum` in `venv\Lib\site-packages\pandas\tests\resample\test_resample_api.py`


**Parameters**:

- `cases`

- `a_mean`

- `a_sum`

- `agg`



**Dependencies**:

- parametrize

- aggregate

- concat

- assert_frame_equal




---


### PARAMETRIZE agg

**Handler**: `test_agg_dict_of_dict_specificationerror` in `venv\Lib\site-packages\pandas\tests\resample\test_resample_api.py`


**Parameters**:

- `cases`

- `agg`



**Dependencies**:

- raises

- parametrize

- aggregate




---


### PARAMETRIZE agg

**Handler**: `test_agg_with_lambda` in `venv\Lib\site-packages\pandas\tests\resample\test_resample_api.py`


**Parameters**:

- `cases`

- `agg`



**Dependencies**:

- parametrize

- apply

- assert_frame_equal

- concat

- std

- assert_produces_warning

- NamedAgg

- sum

- agg




---


### PARAMETRIZE agg

**Handler**: `test_agg_no_column` in `venv\Lib\site-packages\pandas\tests\resample\test_resample_api.py`


**Parameters**:

- `cases`

- `agg`



**Dependencies**:

- NamedAgg

- raises

- parametrize

- agg




---


### PARAMETRIZE cols, agg

**Handler**: `test_agg_specificationerror_nested` in `venv\Lib\site-packages\pandas\tests\resample\test_resample_api.py`


**Parameters**:

- `cases`

- `cols`

- `agg`

- `a_sum`

- `a_std`

- `b_mean`

- `b_std`



**Dependencies**:

- parametrize

- assert_frame_equal

- concat

- from_tuples

- agg




---


### PARAMETRIZE agg

**Handler**: `test_agg_specificationerror_series` in `venv\Lib\site-packages\pandas\tests\resample\test_resample_api.py`


**Parameters**:

- `cases`

- `agg`



**Dependencies**:

- raises

- parametrize

- agg




---


### PARAMETRIZE func

**Handler**: `test_multi_agg_axis_1_raises` in `venv\Lib\site-packages\pandas\tests\resample\test_resample_api.py`


**Parameters**:

- `func`



**Dependencies**:

- raises

- parametrize

- resample

- datetime

- DataFrame

- random

- default_rng

- list

- date_range

- assert_produces_warning

- agg




---


### PARAMETRIZE col_name

**Handler**: `test_agg_with_datetime_index_list_agg_func` in `venv\Lib\site-packages\pandas\tests\resample\test_resample_api.py`


**Parameters**:

- `col_name`



**Dependencies**:

- parametrize

- resample

- assert_frame_equal

- DataFrame

- aggregate

- list

- range

- date_range

- MultiIndex




---


### PARAMETRIZE start,end,freq,data,resample_freq,origin,closed,exp_data,exp_end,exp_periods

**Handler**: `test_end_and_end_day_origin` in `venv\Lib\site-packages\pandas\tests\resample\test_resample_api.py`


**Parameters**:

- `start`

- `end`

- `freq`

- `data`

- `resample_freq`

- `origin`

- `closed`

- `exp_data`

- `exp_end`

- `exp_periods`



**Dependencies**:

- parametrize

- resample

- date_range

- Series

- assert_series_equal

- sum




---


### PARAMETRIZE method, numeric_only, expected_data

**Handler**: `test_frame_downsample_method` in `venv\Lib\site-packages\pandas\tests\resample\test_resample_api.py`


**Parameters**:

- `method`

- `numeric_only`

- `expected_data`

- `using_infer_string`



**Dependencies**:

- getattr

- parametrize

- resample

- raises

- assert_frame_equal

- DataFrame

- escape

- date_range

- isinstance

- func




---


### PARAMETRIZE method, numeric_only, expected_data

**Handler**: `test_series_downsample_method` in `venv\Lib\site-packages\pandas\tests\resample\test_resample_api.py`


**Parameters**:

- `method`

- `numeric_only`

- `expected_data`

- `using_infer_string`



**Dependencies**:

- getattr

- parametrize

- resample

- raises

- escape

- date_range

- Series

- func

- assert_series_equal




---


### PARAMETRIZE method, raises

**Handler**: `test_args_kwargs_depr` in `venv\Lib\site-packages\pandas\tests\resample\test_resample_api.py`


**Parameters**:

- `method`

- `raises`



**Dependencies**:

- getattr

- parametrize

- resample

- raises

- date_range

- Series

- func

- assert_produces_warning




---


### PARAMETRIZE converter

**Handler**: `test_float_int_deprecated` in `venv\Lib\site-packages\pandas\tests\series\test_api.py`


**Parameters**:

- `converter`



**Dependencies**:

- converter

- parametrize

- Series

- assert_produces_warning




---


### PARAMETRIZE index

**Handler**: `test_index_tab_completion` in `venv\Lib\site-packages\pandas\tests\series\test_api.py`


**Parameters**:

- `index`



**Dependencies**:

- dir

- parametrize

- unique

- isidentifier

- Index

- period_range

- arange

- str

- enumerate

- timedelta_range

- list

- Series

- date_range

- isinstance

- range

- from_tuples

- zip




---


### PARAMETRIZE ser

**Handler**: `test_not_hashable` in `venv\Lib\site-packages\pandas\tests\series\test_api.py`


**Parameters**:

- `ser`



**Dependencies**:

- raises

- parametrize

- Series

- hash




---


### PARAMETRIZE dtype

**Handler**: `test_empty_method_full_series` in `venv\Lib\site-packages\pandas\tests\series\test_api.py`


**Parameters**:

- `dtype`



**Dependencies**:

- parametrize

- Series




---


### PARAMETRIZE dtype

**Handler**: `test_integer_series_size` in `venv\Lib\site-packages\pandas\tests\series\test_api.py`


**Parameters**:

- `dtype`



**Dependencies**:

- range

- parametrize

- Series




---


### PARAMETRIZE op

**Handler**: `test_datetime_series_no_datelike_attrs` in `venv\Lib\site-packages\pandas\tests\series\test_api.py`


**Parameters**:

- `op`

- `datetime_series`



**Dependencies**:

- getattr

- raises

- parametrize




---


### FILTERWARNINGS ignore:Downcasting object dtype arrays:FutureWarning

**Handler**: `test_numeric_only` in `venv\Lib\site-packages\pandas\tests\series\test_api.py`


**Parameters**:

- `kernel`

- `has_numeric_only`

- `dtype`



**Dependencies**:

- getattr

- filterwarnings

- parametrize

- raises

- method

- Series

- isinstance

- assert_series_equal




---


### PARAMETRIZE dtype

**Handler**: `test_api_per_dtype` in `venv\Lib\site-packages\pandas\tests\strings\test_api.py`


**Parameters**:

- `index_or_series`

- `dtype`

- `any_skipna_inferred_dtype`



**Dependencies**:

- raises

- parametrize

- isinstance

- hasattr

- box




---


### PARAMETRIZE dtype

**Handler**: `test_api_per_method` in `venv\Lib\site-packages\pandas\tests\strings\test_api.py`


**Parameters**:

- `index_or_series`

- `dtype`

- `any_allowed_skipna_inferred_dtype`

- `any_string_method`

- `request`

- `using_infer_string`



**Dependencies**:

- getattr

- parametrize

- raises

- repr

- option_context

- method

- xfail

- get

- applymarker

- box




---


### PARAMETRIZE func

**Handler**: `test_multi_axis_1_raises` in `venv\Lib\site-packages\pandas\tests\window\test_api.py`


**Parameters**:

- `func`



**Dependencies**:

- raises

- parametrize

- rolling

- DataFrame

- assert_produces_warning

- agg




---


### PARAMETRIZE func,window_size,expected_vals

**Handler**: `test_multiple_agg_funcs` in `venv\Lib\site-packages\pandas\tests\window\test_api.py`


**Parameters**:

- `func`

- `window_size`

- `expected_vals`



**Dependencies**:

- getattr

- parametrize

- assert_frame_equal

- groupby

- DataFrame

- from_tuples

- agg

- f




---


### PARAMETRIZE first

**Handler**: `test_view_between_datetimelike` in `venv\Lib\site-packages\pandas\tests\series\methods\test_view.py`


**Parameters**:

- `first`

- `second`

- `box`



**Dependencies**:

- parametrize

- asarray

- date_range

- view

- assert_numpy_array_equal

- box




---


### PARAMETRIZE in_frame

**Handler**: `test_concat` in `venv\Lib\site-packages\pandas\tests\extension\base\reshaping.py`


**Parameters**:

- `data`

- `in_frame`



**Dependencies**:

- parametrize

- DataFrame

- concat

- Series

- isinstance

- len

- hasattr




---


### PARAMETRIZE in_frame

**Handler**: `test_concat_all_na_block` in `venv\Lib\site-packages\pandas\tests\extension\base\reshaping.py`


**Parameters**:

- `data_missing`

- `in_frame`



**Dependencies**:

- parametrize

- assert_frame_equal

- DataFrame

- concat

- take

- Series

- assert_series_equal




---


### FILTERWARNINGS ignore:The previous implementation of stack is deprecated

**Handler**: `test_stack` in `venv\Lib\site-packages\pandas\tests\extension\base\reshaping.py`


**Parameters**:

- `data`

- `columns`

- `future_stack`



**Dependencies**:

- filterwarnings

- parametrize

- stack

- all

- DataFrame

- assert_equal

- isinstance

- from_tuples

- astype




---


### PARAMETRIZE index

**Handler**: `test_unstack` in `venv\Lib\site-packages\pandas\tests\extension\base\reshaping.py`


**Parameters**:

- `data`

- `index`

- `obj`



**Dependencies**:

- parametrize

- permutations

- droplevel

- all

- assert_frame_equal

- from_product

- DataFrame

- from_iterable

- list

- range

- Series

- isinstance

- to_frame

- type

- unstack

- from_tuples

- astype

- len




---


### PARAMETRIZE cons

**Handler**: `test_datetimeindex` in `venv\Lib\site-packages\pandas\tests\copy_view\index\test_datetimeindex.py`


**Parameters**:

- `using_copy_on_write`

- `cons`



**Dependencies**:

- assert_index_equal

- parametrize

- copy

- date_range

- Series

- cons

- DatetimeIndex

- Timestamp




---


### PARAMETRIZE func

**Handler**: `test_index_ops` in `venv\Lib\site-packages\pandas\tests\copy_view\index\test_index.py`


**Parameters**:

- `using_copy_on_write`

- `func`

- `request`



**Dependencies**:

- _getitem_slice

- assert_index_equal

- parametrize

- slice

- delete

- index_view

- _shallow_copy

- copy

- take

- view

- rename

- func

- astype

- repeat




---


### PARAMETRIZE cons

**Handler**: `test_periodindex` in `venv\Lib\site-packages\pandas\tests\copy_view\index\test_periodindex.py`


**Parameters**:

- `using_copy_on_write`

- `cons`



**Dependencies**:

- assert_index_equal

- parametrize

- PeriodIndex

- period_range

- Period

- copy

- Series

- cons




---


### PARAMETRIZE cons

**Handler**: `test_timedeltaindex` in `venv\Lib\site-packages\pandas\tests\copy_view\index\test_timedeltaindex.py`


**Parameters**:

- `using_copy_on_write`

- `cons`



**Dependencies**:

- assert_index_equal

- parametrize

- Timedelta

- copy

- timedelta_range

- TimedeltaIndex

- Series

- cons




---


### PARAMETRIZE new_categories

**Handler**: `test_rename_categories_wrong_length_raises` in `venv\Lib\site-packages\pandas\tests\arrays\categorical\test_api.py`


**Parameters**:

- `new_categories`



**Dependencies**:

- raises

- parametrize

- rename_categories

- Categorical




---


### PARAMETRIZE new_categories

**Handler**: `test_reorder_categories_raises` in `venv\Lib\site-packages\pandas\tests\arrays\categorical\test_api.py`


**Parameters**:

- `new_categories`



**Dependencies**:

- raises

- parametrize

- reorder_categories

- Categorical




---


### PARAMETRIZE values, categories, new_categories

**Handler**: `test_set_categories_many` in `venv\Lib\site-packages\pandas\tests\arrays\categorical\test_api.py`


**Parameters**:

- `values`

- `categories`

- `new_categories`

- `ordered`



**Dependencies**:

- set_categories

- parametrize

- assert_categorical_equal

- Categorical




---


### PARAMETRIZE removals

**Handler**: `test_remove_categories_raises` in `venv\Lib\site-packages\pandas\tests\arrays\categorical\test_api.py`


**Parameters**:

- `removals`



**Dependencies**:

- raises

- parametrize

- Categorical

- escape

- remove_categories




---


### PARAMETRIZE codes, old, new, expected

**Handler**: `test_recode_to_categories` in `venv\Lib\site-packages\pandas\tests\arrays\categorical\test_api.py`


**Parameters**:

- `codes`

- `old`

- `new`

- `expected`



**Dependencies**:

- parametrize

- Index

- asanyarray

- recode_for_categories

- assert_numpy_array_equal




---


### PARAMETRIZE name

**Handler**: `test_import_lazy_import` in `venv\Lib\site-packages\numpy\tests\test_public_api.py`


**Parameters**:

- `name`



**Dependencies**:

- skipif

- dir

- parametrize

- check_output




---


### FILTERWARNINGS ignore:numpy.core(\.\w+)? is deprecated:DeprecationWarning

**Handler**: `test_core_shims_coherence` in `venv\Lib\site-packages\numpy\tests\test_public_api.py`



**Dependencies**:

- dir

- filterwarnings

- getattr

- __import__

- ismodule

- startswith




---


### PARAMETRIZE array

**Handler**: `test_array_impossible_casts` in `venv\Lib\site-packages\numpy\_core\tests\test_api.py`


**Parameters**:

- `array`



**Dependencies**:

- array

- rational

- parametrize

- assert_raises




---


### PARAMETRIZE dt

**Handler**: `test_array_astype_to_string_discovery_empty` in `venv\Lib\site-packages\numpy\_core\tests\test_api.py`


**Parameters**:

- `dt`



**Dependencies**:

- parametrize

- can_cast

- array

- dtype

- astype




---


### PARAMETRIZE dt

**Handler**: `test_array_astype_to_void` in `venv\Lib\site-packages\numpy\_core\tests\test_api.py`


**Parameters**:

- `dt`



**Dependencies**:

- dtype

- parametrize

- array

- astype




---


### PARAMETRIZE t

**Handler**: `test_array_astype_warning` in `venv\Lib\site-packages\numpy\_core\tests\test_api.py`


**Parameters**:

- `t`



**Dependencies**:

- array

- assert_warns

- parametrize




---


### PARAMETRIZE str_type

**Handler**: `test_string_to_complex_cast` in `venv\Lib\site-packages\numpy\_core\tests\test_api.py`


**Parameters**:

- `str_type`

- `scalar_type`



**Dependencies**:

- parametrize

- array

- zeros

- astype

- scalar_type




---


### PARAMETRIZE dtype

**Handler**: `test_none_to_nan_cast` in `venv\Lib\site-packages\numpy\_core\tests\test_api.py`


**Parameters**:

- `dtype`



**Dependencies**:

- parametrize

- isnan

- array

- zeros

- astype




---


### PARAMETRIZE kind

**Handler**: `test_dtypes_kind` in `venv\Lib\site-packages\numpy\_core\tests\test_array_api_info.py`


**Parameters**:

- `kind`



**Dependencies**:

- parametrize

- isinstance

- dtypes




---


### NOT_IMPLEMENTED_FOR undirected

**Handler**: `reverse_view` in `venv\Lib\site-packages\networkx\classes\graphviews.py`


**Parameters**:

- `G`



**Dependencies**:

- not_implemented_for

- generic_graph_view




---


### PARAMETRIZE graph

**Handler**: `test_cache_dict_get_set_state` in `venv\Lib\site-packages\networkx\classes\tests\test_reportviews.py`


**Parameters**:

- `graph`



**Dependencies**:

- parametrize

- loads

- dumps

- path_graph

- graph

- is_directed

- deepcopy




---


### PARAMETRIZE multigraph

**Handler**: `test_multigraph_filtered_edges` in `venv\Lib\site-packages\networkx\classes\tests\test_subgraphviews.py`


**Parameters**:

- `multigraph`



**Dependencies**:

- has_edge

- parametrize

- edge_subgraph

- multigraph




---


### PARAMETRIZE target,shape_repr,test_shape

**Handler**: `test_check_shape` in `venv\Lib\site-packages\matplotlib\tests\test_api.py`


**Parameters**:

- `target`

- `shape_repr`

- `test_shape`



**Dependencies**:

- raises

- parametrize

- escape

- check_shape

- zeros

- len



**Returns**: None


---


### DELETE_PARAMETER 3.0

**Handler**: `func1` in `venv\Lib\site-packages\matplotlib\tests\test_api.py`


**Parameters**:

- `foo`



**Dependencies**:

- delete_parameter



**Returns**: None


---


### DELETE_PARAMETER 3.0

**Handler**: `func2` in `venv\Lib\site-packages\matplotlib\tests\test_api.py`



**Dependencies**:

- delete_parameter



**Returns**: None


---


### MAKE_KEYWORD_ONLY 3.0

**Handler**: `func` in `venv\Lib\site-packages\matplotlib\tests\test_api.py`


**Parameters**:

- `pre`

- `arg`

- `post`



**Dependencies**:

- make_keyword_only



**Returns**: None


---


### DEPRECATED 1

**Handler**: `f` in `venv\Lib\site-packages\matplotlib\tests\test_api.py`



**Dependencies**:

- deprecated



**Returns**: None


---


### DEPRECATED 0.0.0

**Handler**: `f` in `venv\Lib\site-packages\matplotlib\tests\test_api.py`


**Parameters**:

- `cls`



**Dependencies**:

- deprecated



**Returns**: None


---



## Error Handling

Common HTTP status codes used:
- `200 OK` - Successful request
- `400 Bad Request` - Invalid request parameters
- `401 Unauthorized` - Authentication required
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Server error

## Rate Limiting

API requests may be rate-limited. Check response headers:
- `X-RateLimit-Limit` - Maximum requests per window
- `X-RateLimit-Remaining` - Remaining requests in window
- `X-RateLimit-Reset` - Window reset time

## Authentication


Authentication details not detected in code analysis.


---

*Generated automatically by Deepflow*