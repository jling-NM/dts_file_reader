from dts_file_reader import slice

reader = slice.Reader()

for file_indx in range(1, 21):
    print('\n\n/home/josef/projects/amayer/dtsViewer/data/HYGE_20221130_test'+str(file_indx)+'_01/HYGE_20221130_test'+str(file_indx)+'_01.dts')

    data = reader.parse('/home/josef/projects/amayer/dtsViewer/data/HYGE_20221130_test'+str(file_indx)+'_01/HYGE_20221130_test'+str(file_indx)+'_01.dts')
    head_summary = data[0].get_channel_summary('head')
    print(head_summary)


    head_resultant = slice.get_resultant(data, (0, 1, 2))
    head_resultant_summary = slice.get_data_summary('head', data[0].meta_data.sample_rate_hz, head_resultant)
    print('head_resultant_summary\n', head_resultant_summary)

    machine_resultant = slice.get_resultant(data, (6, 7, 8))
    machine_resultant_summary = slice.get_data_summary('machine', data[8].meta_data.sample_rate_hz, machine_resultant)
    print('machine_resultant_summary\n', machine_resultant_summary)

    exit()
