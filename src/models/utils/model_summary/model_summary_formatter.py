"""Formatting model summary."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import pandas as pd

LOGGER = logging.getLogger(__name__)

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 10000)


def round_value(value, binary=False):
    """Round numbers."""
    divisor = 1024. if binary else 1000.
    if value // divisor**4 > 0:
        return str(round(value / divisor**4, 2)) + ' T'
    if value // divisor**3 > 0:
        return str(round(value / divisor**3, 2)) + ' G'
    if value // divisor**2 > 0:
        return str(round(value / divisor**2, 2)) + ' M'
    if value // divisor > 0:
        return str(round(value / divisor, 2)) + ' K'
    return str(value)


def format_model_summary(collected_nodes):
    """Format model summary from collected nodes."""
    data = list()
    for node in collected_nodes:
        name = node.name
        input_shape = '[' + ','.join(['{:>3d}'] * len(node.input_shape)).format(
            *node.input_shape) + ']'
        output_shape = '[' + ','.join(['{:>3d}'] * len(node.output_shape)).format(
            *node.output_shape) + ']'
        parameter_quantity = node.parameter_quantity
        inference_memory = node.inference_memory
        madd = node.madd
        flops = node.flops
        mread, mwrite = node.memory
        duration = node.duration
        data.append([name, input_shape, output_shape, parameter_quantity,
                     inference_memory, madd, flops, mread, mwrite, duration])
    summary_df = pd.DataFrame(data)
    summary_df.columns = ['module name', 'input shape', 'output shape',
                          'params', 'memory(MB)', 'MAdd', 'FLOPs',
                          'MemRead(B)', 'MemWrite(B)', 'duration(s)']
    summary_df['duration[%]'] = (summary_df['duration(s)'] /
                                 (summary_df['duration(s)'].sum() + 1e-7))
    summary_df['MemR+W(B)'] = (summary_df['MemRead(B)'] +
                               summary_df['MemWrite(B)'])
    total_input_shape = summary_df.iloc[0]['input shape']
    total_output_shape = summary_df.iloc[-1]['output shape']
    total_parameters_quantity = summary_df['params'].sum()
    total_memory = summary_df['memory(MB)'].sum()
    total_operation_quantity = summary_df['MAdd'].sum()
    total_flops = summary_df['FLOPs'].sum()
    total_duration = summary_df['duration(s)'].sum()
    total_duration_percentage = summary_df['duration[%]'].sum()
    total_mread = summary_df['MemRead(B)'].sum()
    total_mwrite = summary_df['MemWrite(B)'].sum()
    total_memrw = summary_df['MemR+W(B)'].sum()

    # Add Total row
    total_df = pd.Series([total_input_shape, total_output_shape,
                          total_parameters_quantity, total_memory,
                          total_operation_quantity, total_flops,
                          total_mread, total_mwrite, total_duration,
                          total_duration_percentage, total_memrw],
                         index=['input shape', 'output shape',
                                'params', 'memory(MB)', 'MAdd', 'FLOPs',
                                'MemRead(B)', 'MemWrite(B)', 'duration(s)',
                                'duration[%]', 'MemR+W(B)'],
                         name='total')
    summary_df = summary_df.append(total_df)

    summary_df = summary_df.fillna(' ')
    summary_df['memory(MB)'] = summary_df['memory(MB)'].apply('{:.2f}'.format)
    summary_df['duration[%]'] = summary_df['duration[%]'].apply(
        '{:.2%}'.format)
    summary_df['MAdd'] = summary_df['MAdd'].apply('{:,}'.format)
    summary_df['FLOPs'] = summary_df['FLOPs'].apply('{:,}'.format)

    LOGGER.info('Total params: {:,}'.format(total_parameters_quantity))
    LOGGER.info('Total memory: {:.2f} MB'.format(total_memory))
    LOGGER.info('Total MAdd: %s MAdd', round_value(total_operation_quantity))
    LOGGER.info('Total FLOPs: %s FLOPs', round_value(total_flops))
    LOGGER.info('Total MemRead: %s B', round_value(total_mread, True))
    LOGGER.info('Total MemWrite: %s B', round_value(total_mwrite, True))
    LOGGER.info('Total MemR+W: %s B', round_value(total_memrw, True))

    return summary_df
