# API Documentation

**Project**: deepflow
**Generated**: 2025-08-23
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

- parametrize

- method

- asarray

- copy

- shares_memory

- assert_series_equal

- raises

- array

- get_array

- Series




---


### PARAMETRIZE method

**Handler**: `test_dataframe_values` in `venv\Lib\site-packages\pandas\tests\copy_view\test_array.py`


**Parameters**:

- `using_copy_on_write`

- `using_array_manager`

- `method`



**Dependencies**:

- DataFrame

- parametrize

- assert_frame_equal

- method

- asarray

- copy

- shares_memory

- raises

- array

- get_array




---


### PARAMETRIZE order

**Handler**: `test_ravel_read_only` in `venv\Lib\site-packages\pandas\tests\copy_view\test_array.py`


**Parameters**:

- `using_copy_on_write`

- `order`



**Dependencies**:

- assert_produces_warning

- parametrize

- shares_memory

- ravel

- get_array

- Series




---


### PARAMETRIZE dtype

**Handler**: `test_astype_avoids_copy` in `venv\Lib\site-packages\pandas\tests\copy_view\test_astype.py`


**Parameters**:

- `using_copy_on_write`

- `dtype`

- `new_dtype`



**Dependencies**:

- DataFrame

- parametrize

- assert_frame_equal

- copy

- importorskip

- shares_memory

- astype

- get_array




---


### PARAMETRIZE dtype

**Handler**: `test_astype_different_target_dtype` in `venv\Lib\site-packages\pandas\tests\copy_view\test_astype.py`


**Parameters**:

- `using_copy_on_write`

- `dtype`



**Dependencies**:

- DataFrame

- parametrize

- assert_frame_equal

- copy

- importorskip

- shares_memory

- astype

- get_array

- _has_no_reference




---


### PARAMETRIZE dtype, new_dtype

**Handler**: `test_astype_string_and_object` in `venv\Lib\site-packages\pandas\tests\copy_view\test_astype.py`


**Parameters**:

- `using_copy_on_write`

- `dtype`

- `new_dtype`



**Dependencies**:

- DataFrame

- parametrize

- assert_frame_equal

- copy

- shares_memory

- astype

- get_array




---


### PARAMETRIZE dtype, new_dtype

**Handler**: `test_astype_string_and_object_update_original` in `venv\Lib\site-packages\pandas\tests\copy_view\test_astype.py`


**Parameters**:

- `using_copy_on_write`

- `dtype`

- `new_dtype`



**Dependencies**:

- DataFrame

- parametrize

- assert_frame_equal

- copy

- shares_memory

- astype

- get_array




---


### PARAMETRIZE func, args

**Handler**: `test_methods_iloc_getitem_item_cache` in `venv\Lib\site-packages\pandas\tests\copy_view\test_chained_assignment_deprecation.py`


**Parameters**:

- `func`

- `args`

- `using_copy_on_write`

- `warn_copy_on_write`



**Dependencies**:

- DataFrame

- parametrize

- raises_chained_assignment_error

- assert_cow_warning

- copy

- getattr




---


### PARAMETRIZE indexer

**Handler**: `test_series_setitem` in `venv\Lib\site-packages\pandas\tests\copy_view\test_chained_assignment_deprecation.py`


**Parameters**:

- `indexer`

- `using_copy_on_write`

- `warn_copy_on_write`



**Dependencies**:

- DataFrame

- warns

- parametrize

- len

- slice

- array




---


### FILTERWARNINGS ignore::pandas.errors.SettingWithCopyWarning

**Handler**: `test_frame_setitem` in `venv\Lib\site-packages\pandas\tests\copy_view\test_chained_assignment_deprecation.py`


**Parameters**:

- `indexer`

- `using_copy_on_write`



**Dependencies**:

- DataFrame

- parametrize

- raises_chained_assignment_error

- filterwarnings

- option_context

- slice

- array




---


### PARAMETRIZE dtype

**Handler**: `test_series_from_series` in `venv\Lib\site-packages\pandas\tests\copy_view\test_constructors.py`


**Parameters**:

- `dtype`

- `using_copy_on_write`

- `warn_copy_on_write`



**Dependencies**:

- parametrize

- assert_cow_warning

- shares_memory

- has_reference

- get_array

- Series




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

- assert_produces_warning

- parametrize

- copy

- shares_memory

- assert_series_equal

- RangeIndex

- getattr

- get_array

- array

- Series




---


### PARAMETRIZE copy

**Handler**: `test_series_from_array_different_dtype` in `venv\Lib\site-packages\pandas\tests\copy_view\test_constructors.py`


**Parameters**:

- `using_copy_on_write`

- `copy`



**Dependencies**:

- parametrize

- shares_memory

- get_array

- array

- Series




---


### PARAMETRIZE idx

**Handler**: `test_series_from_index` in `venv\Lib\site-packages\pandas\tests\copy_view\test_constructors.py`


**Parameters**:

- `using_copy_on_write`

- `idx`



**Dependencies**:

- Timestamp

- assert_index_equal

- parametrize

- Period

- TimedeltaIndex

- Index

- Timedelta

- copy

- shares_memory

- DatetimeIndex

- PeriodIndex

- get_array

- Series

- _has_no_reference




---


### FILTERWARNINGS ignore:Setting a value on a view:FutureWarning

**Handler**: `test_series_from_block_manager` in `venv\Lib\site-packages\pandas\tests\copy_view\test_constructors.py`


**Parameters**:

- `using_copy_on_write`

- `idx`

- `dtype`

- `fastpath`



**Dependencies**:

- assert_produces_warning

- parametrize

- filterwarnings

- copy

- shares_memory

- assert_series_equal

- RangeIndex

- get_array

- Series

- _has_no_reference




---


### PARAMETRIZE use_mgr

**Handler**: `test_dataframe_constructor_mgr_or_df` in `venv\Lib\site-packages\pandas\tests\copy_view\test_constructors.py`


**Parameters**:

- `using_copy_on_write`

- `warn_copy_on_write`

- `columns`

- `use_mgr`



**Dependencies**:

- DataFrame

- assert_produces_warning

- parametrize

- assert_frame_equal

- assert_cow_warning

- copy

- shares_memory

- get_array




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

- DataFrame

- parametrize

- assert_frame_equal

- assert_cow_warning

- copy

- shares_memory

- assert_series_equal

- get_array

- Series




---


### PARAMETRIZE dtype

**Handler**: `test_dataframe_from_dict_of_series_with_reindex` in `venv\Lib\site-packages\pandas\tests\copy_view\test_constructors.py`


**Parameters**:

- `dtype`



**Dependencies**:

- DataFrame

- parametrize

- shares_memory

- get_array

- Series




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

- cons

- DataFrame

- parametrize

- assert_cow_warning

- copy

- shares_memory

- assert_equal

- get_array

- _has_no_reference




---


### PARAMETRIZE cons

**Handler**: `test_dataframe_from_series_or_index_different_dtype` in `venv\Lib\site-packages\pandas\tests\copy_view\test_constructors.py`


**Parameters**:

- `using_copy_on_write`

- `cons`



**Dependencies**:

- cons

- DataFrame

- parametrize

- shares_memory

- get_array

- _has_no_reference




---


### PARAMETRIZE index

**Handler**: `test_dataframe_from_dict_of_series_with_dtype` in `venv\Lib\site-packages\pandas\tests\copy_view\test_constructors.py`


**Parameters**:

- `index`



**Dependencies**:

- DataFrame

- parametrize

- shares_memory

- get_array

- Series




---


### PARAMETRIZE copy

**Handler**: `test_frame_from_numpy_array` in `venv\Lib\site-packages\pandas\tests\copy_view\test_constructors.py`


**Parameters**:

- `using_copy_on_write`

- `copy`

- `using_array_manager`



**Dependencies**:

- DataFrame

- parametrize

- shares_memory

- get_array

- array




---


### PARAMETRIZE copy

**Handler**: `test_concat_copy_keyword` in `venv\Lib\site-packages\pandas\tests\copy_view\test_functions.py`


**Parameters**:

- `using_copy_on_write`

- `copy`



**Dependencies**:

- DataFrame

- parametrize

- shares_memory

- get_array

- concat




---


### PARAMETRIZE func

**Handler**: `test_merge_on_key` in `venv\Lib\site-packages\pandas\tests\copy_view\test_functions.py`


**Parameters**:

- `using_copy_on_write`

- `func`



**Dependencies**:

- DataFrame

- parametrize

- assert_frame_equal

- merge

- func

- copy

- shares_memory

- get_array

- Series




---


### PARAMETRIZE func, how

**Handler**: `test_merge_on_key_enlarging_one` in `venv\Lib\site-packages\pandas\tests\copy_view\test_functions.py`


**Parameters**:

- `using_copy_on_write`

- `func`

- `how`



**Dependencies**:

- DataFrame

- parametrize

- assert_frame_equal

- merge

- func

- copy

- shares_memory

- get_array

- Series

- _has_no_reference




---


### PARAMETRIZE copy

**Handler**: `test_merge_copy_keyword` in `venv\Lib\site-packages\pandas\tests\copy_view\test_functions.py`


**Parameters**:

- `using_copy_on_write`

- `copy`



**Dependencies**:

- DataFrame

- parametrize

- merge

- shares_memory

- get_array




---


### PARAMETRIZE dtype

**Handler**: `test_join_on_key` in `venv\Lib\site-packages\pandas\tests\copy_view\test_functions.py`


**Parameters**:

- `dtype`

- `using_copy_on_write`



**Dependencies**:

- DataFrame

- parametrize

- assert_frame_equal

- Index

- copy

- shares_memory

- join

- get_array




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

- DataFrame

- assert_produces_warning

- parametrize

- _verify_integrity

- assert_frame_equal

- assert_cow_warning

- copy

- shares_memory

- option_context

- get_array

- array




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

- DataFrame

- parametrize

- range

- assert_frame_equal

- isinstance

- assert_cow_warning

- copy

- slice

- array




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

- DataFrame

- parametrize

- range

- assert_frame_equal

- isinstance

- assert_cow_warning

- copy

- slice

- array




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

- DataFrame

- assert_produces_warning

- parametrize

- indexer_si

- assert_frame_equal

- skip

- range

- isinstance

- assert_cow_warning

- copy

- option_context

- slice

- array




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

- DataFrame

- assert_produces_warning

- parametrize

- _verify_integrity

- assert_frame_equal

- range

- assert_cow_warning

- copy

- option_context

- array




---


### PARAMETRIZE dtype

**Handler**: `test_subset_set_columns` in `venv\Lib\site-packages\pandas\tests\copy_view\test_indexing.py`


**Parameters**:

- `backend`

- `using_copy_on_write`

- `warn_copy_on_write`

- `dtype`



**Dependencies**:

- DataFrame

- assert_produces_warning

- parametrize

- _verify_integrity

- assert_frame_equal

- all

- range

- copy

- option_context

- astype

- array

- _has_no_reference




---


### PARAMETRIZE indexer

**Handler**: `test_subset_set_with_column_indexer` in `venv\Lib\site-packages\pandas\tests\copy_view\test_indexing.py`


**Parameters**:

- `backend`

- `indexer`

- `using_copy_on_write`

- `warn_copy_on_write`



**Dependencies**:

- DataFrame

- parametrize

- _verify_integrity

- assert_frame_equal

- range

- assert_cow_warning

- copy

- option_context

- slice

- array




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

- DataFrame

- parametrize

- assert_frame_equal

- method

- endswith

- assert_cow_warning

- copy

- array




---


### PARAMETRIZE dtype

**Handler**: `test_subset_chained_getitem_column` in `venv\Lib\site-packages\pandas\tests\copy_view\test_indexing.py`


**Parameters**:

- `backend`

- `dtype`

- `using_copy_on_write`

- `warn_copy_on_write`



**Dependencies**:

- DataFrame

- parametrize

- assert_frame_equal

- assert_cow_warning

- copy

- _clear_item_cache

- assert_series_equal

- array

- Series




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

- method

- assert_cow_warning

- copy

- assert_series_equal

- Series




---


### PARAMETRIZE method

**Handler**: `test_null_slice` in `venv\Lib\site-packages\pandas\tests\copy_view\test_indexing.py`


**Parameters**:

- `backend`

- `method`

- `using_copy_on_write`

- `warn_copy_on_write`



**Dependencies**:

- DataFrame

- parametrize

- assert_frame_equal

- method

- assert_cow_warning

- copy




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

- method

- assert_cow_warning

- copy

- assert_series_equal

- Series




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

- assert_produces_warning

- parametrize

- indexer_si

- isinstance

- assert_cow_warning

- copy

- assert_series_equal

- slice

- array

- Series




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

- DataFrame

- assert_produces_warning

- parametrize

- assert_frame_equal

- method

- assert_cow_warning

- copy

- assert_series_equal

- option_context




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

- DataFrame

- assert_produces_warning

- parametrize

- assert_frame_equal

- assert_cow_warning

- copy

- shares_memory

- indexer_func

- slice

- get_array




---


### PARAMETRIZE dtype

**Handler**: `test_iset_splits_blocks_inplace` in `venv\Lib\site-packages\pandas\tests\copy_view\test_internals.py`


**Parameters**:

- `using_copy_on_write`

- `locs`

- `arr`

- `dtype`



**Dependencies**:

- DataFrame

- parametrize

- assert_frame_equal

- copy

- iset

- shares_memory

- astype

- enumerate

- get_array

- array

- Series




---


### PARAMETRIZE method

**Handler**: `test_interpolate_no_op` in `venv\Lib\site-packages\pandas\tests\copy_view\test_interp_fillna.py`


**Parameters**:

- `using_copy_on_write`

- `method`



**Dependencies**:

- DataFrame

- assert_produces_warning

- parametrize

- assert_frame_equal

- interpolate

- copy

- shares_memory

- get_array




---


### PARAMETRIZE func

**Handler**: `test_interp_fill_functions` in `venv\Lib\site-packages\pandas\tests\copy_view\test_interp_fillna.py`


**Parameters**:

- `using_copy_on_write`

- `func`



**Dependencies**:

- DataFrame

- parametrize

- assert_frame_equal

- copy

- shares_memory

- getattr

- get_array




---


### PARAMETRIZE func

**Handler**: `test_interpolate_triggers_copy` in `venv\Lib\site-packages\pandas\tests\copy_view\test_interp_fillna.py`


**Parameters**:

- `using_copy_on_write`

- `vals`

- `func`



**Dependencies**:

- DataFrame

- Timestamp

- parametrize

- shares_memory

- getattr

- get_array

- _has_no_reference




---


### PARAMETRIZE vals

**Handler**: `test_interpolate_inplace_no_reference_no_copy` in `venv\Lib\site-packages\pandas\tests\copy_view\test_interp_fillna.py`


**Parameters**:

- `using_copy_on_write`

- `vals`



**Dependencies**:

- DataFrame

- Timestamp

- parametrize

- interpolate

- shares_memory

- get_array

- _has_no_reference




---


### PARAMETRIZE vals

**Handler**: `test_interpolate_inplace_with_refs` in `venv\Lib\site-packages\pandas\tests\copy_view\test_interp_fillna.py`


**Parameters**:

- `using_copy_on_write`

- `vals`

- `warn_copy_on_write`



**Dependencies**:

- DataFrame

- Timestamp

- parametrize

- assert_frame_equal

- interpolate

- assert_cow_warning

- copy

- shares_memory

- get_array

- _has_no_reference




---


### PARAMETRIZE func

**Handler**: `test_interp_fill_functions_inplace` in `venv\Lib\site-packages\pandas\tests\copy_view\test_interp_fillna.py`


**Parameters**:

- `using_copy_on_write`

- `func`

- `warn_copy_on_write`

- `dtype`



**Dependencies**:

- DataFrame

- parametrize

- assert_frame_equal

- assert_cow_warning

- copy

- shares_memory

- getattr

- get_array

- _has_no_reference




---


### PARAMETRIZE downcast

**Handler**: `test_fillna_inplace` in `venv\Lib\site-packages\pandas\tests\copy_view\test_interp_fillna.py`


**Parameters**:

- `using_copy_on_write`

- `downcast`



**Dependencies**:

- DataFrame

- assert_produces_warning

- parametrize

- fillna

- shares_memory

- get_array

- _has_no_reference




---


### PARAMETRIZE func

**Handler**: `test_interpolate_chained_assignment` in `venv\Lib\site-packages\pandas\tests\copy_view\test_interp_fillna.py`


**Parameters**:

- `using_copy_on_write`

- `func`



**Dependencies**:

- DataFrame

- assert_produces_warning

- parametrize

- assert_frame_equal

- raises_chained_assignment_error

- copy

- option_context

- getattr




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

- rename_axis

- get_array

- truncate

- date_range

- assert_produces_warning

- rename

- reindex_like

- set_axis

- startswith

- astype

- tz_convert

- align

- DataFrame

- parametrize

- reindex

- method

- set_flags

- shares_memory

- swapaxes

- infer_objects

- tz_localize

- to_timestamp

- to_period

- period_range




---


### PARAMETRIZE copy

**Handler**: `test_methods_series_copy_keyword` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `request`

- `method`

- `copy`

- `using_copy_on_write`



**Dependencies**:

- rename_axis

- get_array

- truncate

- date_range

- assert_produces_warning

- rename

- reindex_like

- set_axis

- astype

- tz_convert

- align

- Series

- from_arrays

- parametrize

- reindex

- method

- set_flags

- shares_memory

- swapaxes

- infer_objects

- tz_localize

- to_timestamp

- swaplevel

- to_period

- period_range




---


### PARAMETRIZE copy

**Handler**: `test_transpose_copy_keyword` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `copy`

- `using_array_manager`



**Dependencies**:

- DataFrame

- transpose

- parametrize

- shares_memory

- get_array




---


### PARAMETRIZE index

**Handler**: `test_reset_index_series_drop` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `index`



**Dependencies**:

- parametrize

- reset_index

- Index

- copy

- shares_memory

- assert_series_equal

- RangeIndex

- get_array

- Series

- _has_no_reference




---


### PARAMETRIZE index

**Handler**: `test_reindex_rows` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `index`

- `using_copy_on_write`



**Dependencies**:

- DataFrame

- view

- parametrize

- assert_frame_equal

- reindex

- index

- copy

- shares_memory

- list

- get_array




---


### PARAMETRIZE filter_kwargs

**Handler**: `test_filter` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `filter_kwargs`



**Dependencies**:

- DataFrame

- parametrize

- assert_frame_equal

- filter

- copy

- shares_memory

- get_array




---


### PARAMETRIZE func

**Handler**: `test_align_frame` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `func`



**Dependencies**:

- DataFrame

- parametrize

- assert_frame_equal

- func

- copy

- shares_memory

- slice

- get_array

- align




---


### PARAMETRIZE ax

**Handler**: `test_swapaxes_noop` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `ax`



**Dependencies**:

- DataFrame

- assert_produces_warning

- parametrize

- assert_frame_equal

- copy

- shares_memory

- get_array

- swapaxes




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

- DataFrame

- parametrize

- reset_index

- assert_frame_equal

- method

- rename

- assert_cow_warning

- copy

- select_dtypes




---


### PARAMETRIZE obj

**Handler**: `test_to_timestamp` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `obj`



**Dependencies**:

- DataFrame

- parametrize

- Period

- Index

- copy

- shares_memory

- to_timestamp

- assert_equal

- get_array

- Series




---


### PARAMETRIZE obj

**Handler**: `test_to_period` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `obj`



**Dependencies**:

- DataFrame

- Timestamp

- parametrize

- Index

- copy

- shares_memory

- assert_equal

- to_period

- get_array

- Series




---


### PARAMETRIZE axis, val

**Handler**: `test_dropna` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `axis`

- `val`



**Dependencies**:

- DataFrame

- parametrize

- assert_frame_equal

- copy

- shares_memory

- dropna

- get_array




---


### PARAMETRIZE val

**Handler**: `test_dropna_series` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `val`



**Dependencies**:

- parametrize

- copy

- shares_memory

- assert_series_equal

- dropna

- Series




---


### PARAMETRIZE method

**Handler**: `test_head_tail` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `method`

- `using_copy_on_write`

- `warn_copy_on_write`



**Dependencies**:

- DataFrame

- parametrize

- _verify_integrity

- assert_frame_equal

- method

- head

- assert_cow_warning

- copy

- shares_memory

- get_array

- tail




---


### PARAMETRIZE kwargs

**Handler**: `test_truncate` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `kwargs`



**Dependencies**:

- DataFrame

- truncate

- parametrize

- _verify_integrity

- assert_frame_equal

- copy

- shares_memory

- get_array




---


### PARAMETRIZE method

**Handler**: `test_assign_drop_duplicates` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `method`



**Dependencies**:

- DataFrame

- parametrize

- _verify_integrity

- assert_frame_equal

- copy

- shares_memory

- getattr

- get_array




---


### PARAMETRIZE obj

**Handler**: `test_take` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `obj`



**Dependencies**:

- DataFrame

- parametrize

- copy

- shares_memory

- assert_equal

- take

- Series




---


### PARAMETRIZE obj

**Handler**: `test_between_time` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `obj`



**Dependencies**:

- DataFrame

- date_range

- parametrize

- copy

- shares_memory

- between_time

- assert_equal

- Series




---


### PARAMETRIZE obj, kwargs

**Handler**: `test_sort_values` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `obj`

- `kwargs`



**Dependencies**:

- DataFrame

- parametrize

- copy

- shares_memory

- get_array

- assert_equal

- sort_values

- Series




---


### PARAMETRIZE obj, kwargs

**Handler**: `test_sort_values_inplace` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `obj`

- `kwargs`

- `warn_copy_on_write`



**Dependencies**:

- sort_values

- DataFrame

- parametrize

- assert_cow_warning

- copy

- shares_memory

- assert_equal

- get_array

- Series




---


### PARAMETRIZE decimals

**Handler**: `test_round` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `warn_copy_on_write`

- `decimals`



**Dependencies**:

- DataFrame

- parametrize

- assert_frame_equal

- round

- copy

- shares_memory

- get_array




---


### PARAMETRIZE obj

**Handler**: `test_swaplevel` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `obj`



**Dependencies**:

- DataFrame

- parametrize

- copy

- shares_memory

- swaplevel

- from_tuples

- assert_equal

- Series




---


### PARAMETRIZE kwargs

**Handler**: `test_rename_axis` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `kwargs`



**Dependencies**:

- DataFrame

- parametrize

- assert_frame_equal

- rename_axis

- Index

- copy

- shares_memory

- get_array




---


### PARAMETRIZE func, tz

**Handler**: `test_tz_convert_localize` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `func`

- `tz`



**Dependencies**:

- date_range

- parametrize

- copy

- shares_memory

- assert_series_equal

- getattr

- Series




---


### PARAMETRIZE dtype

**Handler**: `test_putmask` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `dtype`

- `warn_copy_on_write`



**Dependencies**:

- DataFrame

- parametrize

- assert_frame_equal

- assert_cow_warning

- copy

- shares_memory

- get_array




---


### PARAMETRIZE dtype

**Handler**: `test_putmask_no_reference` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `dtype`



**Dependencies**:

- DataFrame

- get_array

- parametrize

- shares_memory




---


### PARAMETRIZE dtype

**Handler**: `test_putmask_aligns_rhs_no_reference` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `dtype`



**Dependencies**:

- DataFrame

- get_array

- parametrize

- shares_memory




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

- DataFrame

- assert_produces_warning

- parametrize

- assert_frame_equal

- assert_cow_warning

- copy

- shares_memory

- list

- get_array

- _has_no_reference




---


### PARAMETRIZE dtype

**Handler**: `test_where_mask_noop` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `dtype`

- `func`



**Dependencies**:

- parametrize

- where

- func

- copy

- shares_memory

- assert_series_equal

- mask

- get_array

- Series




---


### PARAMETRIZE dtype

**Handler**: `test_where_mask` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `dtype`

- `func`



**Dependencies**:

- parametrize

- where

- func

- copy

- shares_memory

- assert_series_equal

- mask

- get_array

- Series




---


### PARAMETRIZE dtype, val

**Handler**: `test_where_mask_noop_on_single_column` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `dtype`

- `val`

- `func`



**Dependencies**:

- DataFrame

- parametrize

- assert_frame_equal

- where

- func

- copy

- shares_memory

- mask

- get_array




---


### PARAMETRIZE func

**Handler**: `test_chained_where_mask` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `func`



**Dependencies**:

- DataFrame

- assert_produces_warning

- parametrize

- assert_frame_equal

- raises_chained_assignment_error

- copy

- option_context

- getattr




---


### PARAMETRIZE dtype

**Handler**: `test_isetitem_series` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `dtype`



**Dependencies**:

- DataFrame

- parametrize

- assert_frame_equal

- isetitem

- copy

- shares_memory

- assert_series_equal

- get_array

- array

- Series

- _has_no_reference




---


### PARAMETRIZE key

**Handler**: `test_get` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `warn_copy_on_write`

- `key`



**Dependencies**:

- DataFrame

- assert_produces_warning

- parametrize

- assert_frame_equal

- isinstance

- copy

- shares_memory

- option_context

- get

- get_array




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

- DataFrame

- assert_produces_warning

- parametrize

- xs

- assert_frame_equal

- assert_cow_warning

- copy

- shares_memory

- option_context

- get_array

- array

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

- DataFrame

- transpose

- assert_produces_warning

- parametrize

- xs

- assert_frame_equal

- from_product

- arange

- copy

- shares_memory

- option_context

- list

- reshape

- get_array




---


### PARAMETRIZE copy

**Handler**: `test_transpose` in `venv\Lib\site-packages\pandas\tests\copy_view\test_methods.py`


**Parameters**:

- `using_copy_on_write`

- `copy`

- `using_array_manager`



**Dependencies**:

- DataFrame

- transpose

- parametrize

- assert_frame_equal

- copy

- shares_memory

- get_array




---


### PARAMETRIZE replace_kwargs

**Handler**: `test_replace` in `venv\Lib\site-packages\pandas\tests\copy_view\test_replace.py`


**Parameters**:

- `using_copy_on_write`

- `replace_kwargs`



**Dependencies**:

- DataFrame

- parametrize

- all

- assert_frame_equal

- copy

- shares_memory

- replace

- get_array




---


### PARAMETRIZE to_replace

**Handler**: `test_replace_inplace` in `venv\Lib\site-packages\pandas\tests\copy_view\test_replace.py`


**Parameters**:

- `using_copy_on_write`

- `to_replace`



**Dependencies**:

- DataFrame

- parametrize

- shares_memory

- replace

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

- DataFrame

- parametrize

- assert_cow_warning

- shares_memory

- replace

- get_array

- _has_no_reference




---


### PARAMETRIZE to_replace

**Handler**: `test_replace_inplace_reference_no_op` in `venv\Lib\site-packages\pandas\tests\copy_view\test_replace.py`


**Parameters**:

- `using_copy_on_write`

- `to_replace`



**Dependencies**:

- DataFrame

- parametrize

- shares_memory

- replace

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

- DataFrame

- assert_produces_warning

- parametrize

- assert_frame_equal

- copy

- shares_memory

- Categorical

- replace

- get_array

- _has_no_reference




---


### PARAMETRIZE val

**Handler**: `test_replace_categorical_inplace` in `venv\Lib\site-packages\pandas\tests\copy_view\test_replace.py`


**Parameters**:

- `using_copy_on_write`

- `val`



**Dependencies**:

- DataFrame

- assert_produces_warning

- parametrize

- assert_frame_equal

- shares_memory

- Categorical

- replace

- get_array

- _has_no_reference




---


### PARAMETRIZE val

**Handler**: `test_replace_categorical` in `venv\Lib\site-packages\pandas\tests\copy_view\test_replace.py`


**Parameters**:

- `using_copy_on_write`

- `val`



**Dependencies**:

- DataFrame

- assert_produces_warning

- parametrize

- assert_frame_equal

- copy

- shares_memory

- Categorical

- replace

- get_array

- _has_no_reference




---


### PARAMETRIZE method

**Handler**: `test_masking_inplace` in `venv\Lib\site-packages\pandas\tests\copy_view\test_replace.py`


**Parameters**:

- `using_copy_on_write`

- `method`

- `warn_copy_on_write`



**Dependencies**:

- DataFrame

- parametrize

- assert_frame_equal

- method

- assert_cow_warning

- copy

- shares_memory

- getattr

- get_array

- _has_no_reference




---


### PARAMETRIZE value

**Handler**: `test_replace_object_list_inplace` in `venv\Lib\site-packages\pandas\tests\copy_view\test_replace.py`


**Parameters**:

- `using_copy_on_write`

- `value`



**Dependencies**:

- DataFrame

- parametrize

- shares_memory

- replace

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

- DataFrame

- parametrize

- set_flags

- assert_cow_warning

- may_share_memory




---


### PARAMETRIZE name

**Handler**: `test_deprecations` in `venv\Lib\site-packages\pandas\tests\internals\test_api.py`


**Parameters**:

- `name`



**Dependencies**:

- assert_produces_warning

- parametrize

- getattr




---


### PARAMETRIZE key

**Handler**: `test_select_bad_cols` in `venv\Lib\site-packages\pandas\tests\resample\test_resample_api.py`


**Parameters**:

- `key`

- `test_frame`



**Dependencies**:

- parametrize

- resample

- raises




---


### PARAMETRIZE attr

**Handler**: `test_api_compat_before_use` in `venv\Lib\site-packages\pandas\tests\resample\test_resample_api.py`


**Parameters**:

- `attr`



**Dependencies**:

- date_range

- parametrize

- resample

- arange

- getattr

- len

- mean

- Series




---


### PARAMETRIZE on

**Handler**: `test_transform_frame` in `venv\Lib\site-packages\pandas\tests\resample\test_resample_api.py`


**Parameters**:

- `on`



**Dependencies**:

- DataFrame

- date_range

- parametrize

- reset_index

- resample

- assert_frame_equal

- random

- groupby

- Grouper

- datetime

- transform

- list

- default_rng




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

- func

- groupby

- Grouper

- assert_series_equal




---


### PARAMETRIZE agg

**Handler**: `test_agg_both_mean_std_named_result` in `venv\Lib\site-packages\pandas\tests\resample\test_resample_api.py`


**Parameters**:

- `cases`

- `a_mean`

- `b_std`

- `agg`



**Dependencies**:

- assert_produces_warning

- parametrize

- assert_frame_equal

- aggregate

- NamedAgg

- concat




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

- concat

- assert_frame_equal

- aggregate




---


### PARAMETRIZE agg

**Handler**: `test_agg_dict_of_dict_specificationerror` in `venv\Lib\site-packages\pandas\tests\resample\test_resample_api.py`


**Parameters**:

- `cases`

- `agg`



**Dependencies**:

- aggregate

- parametrize

- raises




---


### PARAMETRIZE agg

**Handler**: `test_agg_with_lambda` in `venv\Lib\site-packages\pandas\tests\resample\test_resample_api.py`


**Parameters**:

- `cases`

- `agg`



**Dependencies**:

- assert_produces_warning

- parametrize

- assert_frame_equal

- NamedAgg

- apply

- std

- agg

- sum

- concat




---


### PARAMETRIZE agg

**Handler**: `test_agg_no_column` in `venv\Lib\site-packages\pandas\tests\resample\test_resample_api.py`


**Parameters**:

- `cases`

- `agg`



**Dependencies**:

- agg

- NamedAgg

- parametrize

- raises




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

- agg

- from_tuples

- concat




---


### PARAMETRIZE agg

**Handler**: `test_agg_specificationerror_series` in `venv\Lib\site-packages\pandas\tests\resample\test_resample_api.py`


**Parameters**:

- `cases`

- `agg`



**Dependencies**:

- agg

- parametrize

- raises




---


### PARAMETRIZE func

**Handler**: `test_multi_agg_axis_1_raises` in `venv\Lib\site-packages\pandas\tests\resample\test_resample_api.py`


**Parameters**:

- `func`



**Dependencies**:

- DataFrame

- date_range

- assert_produces_warning

- parametrize

- resample

- random

- datetime

- agg

- raises

- list

- default_rng




---


### PARAMETRIZE col_name

**Handler**: `test_agg_with_datetime_index_list_agg_func` in `venv\Lib\site-packages\pandas\tests\resample\test_resample_api.py`


**Parameters**:

- `col_name`



**Dependencies**:

- DataFrame

- date_range

- parametrize

- range

- assert_frame_equal

- aggregate

- resample

- MultiIndex

- list




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

- date_range

- parametrize

- resample

- assert_series_equal

- sum

- Series




---


### PARAMETRIZE method, numeric_only, expected_data

**Handler**: `test_frame_downsample_method` in `venv\Lib\site-packages\pandas\tests\resample\test_resample_api.py`


**Parameters**:

- `method`

- `numeric_only`

- `expected_data`

- `using_infer_string`



**Dependencies**:

- DataFrame

- date_range

- parametrize

- resample

- assert_frame_equal

- isinstance

- escape

- func

- getattr

- raises




---


### PARAMETRIZE method, numeric_only, expected_data

**Handler**: `test_series_downsample_method` in `venv\Lib\site-packages\pandas\tests\resample\test_resample_api.py`


**Parameters**:

- `method`

- `numeric_only`

- `expected_data`

- `using_infer_string`



**Dependencies**:

- date_range

- parametrize

- resample

- escape

- func

- assert_series_equal

- getattr

- raises

- Series




---


### PARAMETRIZE method, raises

**Handler**: `test_args_kwargs_depr` in `venv\Lib\site-packages\pandas\tests\resample\test_resample_api.py`


**Parameters**:

- `method`

- `raises`



**Dependencies**:

- date_range

- assert_produces_warning

- parametrize

- resample

- func

- getattr

- raises

- Series




---


### PARAMETRIZE converter

**Handler**: `test_float_int_deprecated` in `venv\Lib\site-packages\pandas\tests\series\test_api.py`


**Parameters**:

- `converter`



**Dependencies**:

- assert_produces_warning

- parametrize

- converter

- Series




---


### PARAMETRIZE index

**Handler**: `test_index_tab_completion` in `venv\Lib\site-packages\pandas\tests\series\test_api.py`


**Parameters**:

- `index`



**Dependencies**:

- date_range

- parametrize

- dir

- range

- timedelta_range

- str

- isinstance

- Index

- unique

- arange

- isidentifier

- enumerate

- from_tuples

- zip

- list

- period_range

- Series




---


### PARAMETRIZE ser

**Handler**: `test_not_hashable` in `venv\Lib\site-packages\pandas\tests\series\test_api.py`


**Parameters**:

- `ser`



**Dependencies**:

- parametrize

- Series

- hash

- raises




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

- parametrize

- range

- Series




---


### PARAMETRIZE op

**Handler**: `test_datetime_series_no_datelike_attrs` in `venv\Lib\site-packages\pandas\tests\series\test_api.py`


**Parameters**:

- `op`

- `datetime_series`



**Dependencies**:

- parametrize

- getattr

- raises




---


### FILTERWARNINGS ignore:Downcasting object dtype arrays:FutureWarning

**Handler**: `test_numeric_only` in `venv\Lib\site-packages\pandas\tests\series\test_api.py`


**Parameters**:

- `kernel`

- `has_numeric_only`

- `dtype`



**Dependencies**:

- parametrize

- filterwarnings

- method

- isinstance

- assert_series_equal

- getattr

- raises

- Series




---


### PARAMETRIZE dtype

**Handler**: `test_api_per_dtype` in `venv\Lib\site-packages\pandas\tests\strings\test_api.py`


**Parameters**:

- `index_or_series`

- `dtype`

- `any_skipna_inferred_dtype`



**Dependencies**:

- parametrize

- isinstance

- hasattr

- raises

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

- parametrize

- method

- applymarker

- option_context

- getattr

- raises

- get

- repr

- xfail

- box




---


### PARAMETRIZE func

**Handler**: `test_multi_axis_1_raises` in `venv\Lib\site-packages\pandas\tests\window\test_api.py`


**Parameters**:

- `func`



**Dependencies**:

- DataFrame

- assert_produces_warning

- parametrize

- rolling

- agg

- raises




---


### PARAMETRIZE func,window_size,expected_vals

**Handler**: `test_multiple_agg_funcs` in `venv\Lib\site-packages\pandas\tests\window\test_api.py`


**Parameters**:

- `func`

- `window_size`

- `expected_vals`



**Dependencies**:

- DataFrame

- parametrize

- assert_frame_equal

- groupby

- agg

- f

- getattr

- from_tuples




---


### PARAMETRIZE first

**Handler**: `test_view_between_datetimelike` in `venv\Lib\site-packages\pandas\tests\series\methods\test_view.py`


**Parameters**:

- `first`

- `second`

- `box`



**Dependencies**:

- date_range

- view

- parametrize

- asarray

- assert_numpy_array_equal

- box




---


### PARAMETRIZE in_frame

**Handler**: `test_concat` in `venv\Lib\site-packages\pandas\tests\extension\base\reshaping.py`


**Parameters**:

- `data`

- `in_frame`



**Dependencies**:

- DataFrame

- parametrize

- isinstance

- hasattr

- len

- concat

- Series




---


### PARAMETRIZE in_frame

**Handler**: `test_concat_all_na_block` in `venv\Lib\site-packages\pandas\tests\extension\base\reshaping.py`


**Parameters**:

- `data_missing`

- `in_frame`



**Dependencies**:

- DataFrame

- parametrize

- assert_frame_equal

- assert_series_equal

- concat

- take

- Series




---


### FILTERWARNINGS ignore:The previous implementation of stack is deprecated

**Handler**: `test_stack` in `venv\Lib\site-packages\pandas\tests\extension\base\reshaping.py`


**Parameters**:

- `data`

- `columns`

- `future_stack`



**Dependencies**:

- DataFrame

- parametrize

- all

- filterwarnings

- isinstance

- stack

- astype

- from_tuples

- assert_equal




---


### PARAMETRIZE index

**Handler**: `test_unstack` in `venv\Lib\site-packages\pandas\tests\extension\base\reshaping.py`


**Parameters**:

- `data`

- `index`

- `obj`



**Dependencies**:

- DataFrame

- permutations

- parametrize

- range

- unstack

- all

- assert_frame_equal

- type

- isinstance

- from_product

- from_iterable

- astype

- droplevel

- from_tuples

- len

- list

- to_frame

- Series




---


### PARAMETRIZE cons

**Handler**: `test_datetimeindex` in `venv\Lib\site-packages\pandas\tests\copy_view\index\test_datetimeindex.py`


**Parameters**:

- `using_copy_on_write`

- `cons`



**Dependencies**:

- cons

- date_range

- Timestamp

- parametrize

- assert_index_equal

- copy

- DatetimeIndex

- Series




---


### PARAMETRIZE func

**Handler**: `test_index_ops` in `venv\Lib\site-packages\pandas\tests\copy_view\index\test_index.py`


**Parameters**:

- `using_copy_on_write`

- `func`

- `request`



**Dependencies**:

- view

- index_view

- parametrize

- assert_index_equal

- repeat

- rename

- func

- delete

- copy

- _getitem_slice

- astype

- slice

- _shallow_copy

- take




---


### PARAMETRIZE cons

**Handler**: `test_periodindex` in `venv\Lib\site-packages\pandas\tests\copy_view\index\test_periodindex.py`


**Parameters**:

- `using_copy_on_write`

- `cons`



**Dependencies**:

- cons

- parametrize

- assert_index_equal

- Period

- copy

- PeriodIndex

- period_range

- Series




---


### PARAMETRIZE cons

**Handler**: `test_timedeltaindex` in `venv\Lib\site-packages\pandas\tests\copy_view\index\test_timedeltaindex.py`


**Parameters**:

- `using_copy_on_write`

- `cons`



**Dependencies**:

- cons

- parametrize

- assert_index_equal

- timedelta_range

- TimedeltaIndex

- Timedelta

- copy

- Series




---


### PARAMETRIZE new_categories

**Handler**: `test_rename_categories_wrong_length_raises` in `venv\Lib\site-packages\pandas\tests\arrays\categorical\test_api.py`


**Parameters**:

- `new_categories`



**Dependencies**:

- rename_categories

- parametrize

- raises

- Categorical




---


### PARAMETRIZE new_categories

**Handler**: `test_reorder_categories_raises` in `venv\Lib\site-packages\pandas\tests\arrays\categorical\test_api.py`


**Parameters**:

- `new_categories`



**Dependencies**:

- reorder_categories

- parametrize

- raises

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

- assert_categorical_equal

- parametrize

- set_categories

- Categorical




---


### PARAMETRIZE removals

**Handler**: `test_remove_categories_raises` in `venv\Lib\site-packages\pandas\tests\arrays\categorical\test_api.py`


**Parameters**:

- `removals`



**Dependencies**:

- parametrize

- remove_categories

- escape

- Categorical

- raises




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

- recode_for_categories

- assert_numpy_array_equal

- asanyarray




---


### PARAMETRIZE name

**Handler**: `test_import_lazy_import` in `venv\Lib\site-packages\numpy\tests\test_public_api.py`


**Parameters**:

- `name`



**Dependencies**:

- check_output

- skipif

- parametrize

- dir




---


### FILTERWARNINGS ignore:numpy.core(\.\w+)? is deprecated:DeprecationWarning

**Handler**: `test_core_shims_coherence` in `venv\Lib\site-packages\numpy\tests\test_public_api.py`



**Dependencies**:

- dir

- filterwarnings

- ismodule

- __import__

- startswith

- getattr




---


### PARAMETRIZE array

**Handler**: `test_array_impossible_casts` in `venv\Lib\site-packages\numpy\_core\tests\test_api.py`


**Parameters**:

- `array`



**Dependencies**:

- rational

- array

- parametrize

- assert_raises




---


### PARAMETRIZE dt

**Handler**: `test_array_astype_to_string_discovery_empty` in `venv\Lib\site-packages\numpy\_core\tests\test_api.py`


**Parameters**:

- `dt`



**Dependencies**:

- dtype

- parametrize

- can_cast

- astype

- array




---


### PARAMETRIZE dt

**Handler**: `test_array_astype_to_void` in `venv\Lib\site-packages\numpy\_core\tests\test_api.py`


**Parameters**:

- `dt`



**Dependencies**:

- array

- astype

- dtype

- parametrize




---


### PARAMETRIZE t

**Handler**: `test_array_astype_warning` in `venv\Lib\site-packages\numpy\_core\tests\test_api.py`


**Parameters**:

- `t`



**Dependencies**:

- assert_warns

- array

- parametrize




---


### PARAMETRIZE str_type

**Handler**: `test_string_to_complex_cast` in `venv\Lib\site-packages\numpy\_core\tests\test_api.py`


**Parameters**:

- `str_type`

- `scalar_type`



**Dependencies**:

- parametrize

- scalar_type

- astype

- array

- zeros




---


### PARAMETRIZE dtype

**Handler**: `test_none_to_nan_cast` in `venv\Lib\site-packages\numpy\_core\tests\test_api.py`


**Parameters**:

- `dtype`



**Dependencies**:

- parametrize

- astype

- array

- isnan

- zeros




---


### PARAMETRIZE kind

**Handler**: `test_dtypes_kind` in `venv\Lib\site-packages\numpy\_core\tests\test_array_api_info.py`


**Parameters**:

- `kind`



**Dependencies**:

- isinstance

- parametrize

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

- is_directed

- loads

- deepcopy

- path_graph

- graph

- dumps




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

- parametrize

- escape

- check_shape

- len

- raises

- zeros



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