import pandas as pd
import re
import os
import numpy as np
import time

def time_convert(sec):
  mins = sec // 60
  sec = sec % 60
  hours = mins // 60
  mins = mins % 60
  print("Time Lapsed = {0}:{1}:{2}".format(int(hours),int(mins),sec))

def cleanup_and_convert_tofloat(val):
    """
    Clean up and convert the string number value to a float
     - Delete any non-numeric character: a-z%()[], except '-'
     - Delete any non-numeric rows
     - Convert to float type 
    """
    new_val = val.astype(str)
    
    if new_val.str.contains('[+-]?[0-9]*\.?[0-9]+m[aAvVWw]\/K', regex = True).any():
        new_val = new_val.str.findall('[+-]?[0-9]*\.?[0-9]+m[aAvVWw]\/K').str.join(", ")
    else:
        new_val = new_val.replace({'[a-zA-Z%()\[\]\s\/:]':'',}, regex=True)
        if new_val.str.contains('-').any() or new_val.str.contains('\+').any or new_val.str.contains('±').any(): #check whether a ranged values or values with + sign exist, for example in efficiency_df: 23.0-25.6%, or in temp coeffs
            new_val = new_val.replace('', np.nan).dropna().reset_index(drop=True)
        else:
            new_val = new_val.replace('', np.nan).dropna().reset_index(drop=True).astype('float')
    return new_val


def check_empty_dataframe(df):
  # Check if the DataFrame is empty
  if df.empty:
    # Create a new DataFrame with NaN values
    df = pd.DataFrame(np.nan, index = range(0,len(df.index)-1), columns=['none'])

  # Return the DataFrame
  return df


def filter_stc_data(df, case, check_table_title_temp_coeff = False):
    
    if case == "electrical_params":
        
        efficiency_df = df.filter(regex=re.compile(pattern_efficiency, re.IGNORECASE), axis=1)
        if len(efficiency_df.columns) > 1: #check if df.filter method returns multiple columns
            for col_name, col_values in efficiency_df.iteritems():
                #check if the dataframe contains number and not containing product number/model type
                # SIMPLIFY THIS LATER? 
                if not type(col_values.iloc[-1]) == str:
                    if re.search(r"^[0-9]+%?\.?[0-9]*%?-*±?[0-9]*%?\.?[0-9]*%?", col_values.iloc[-1].astype(str)):
                        continue
                    else: 
                        efficiency_df = efficiency_df.drop(columns=[col_name])
                              
                if re.search(r"^[0-9]+%?\.?[0-9]*%?-*±?[0-9]*%?\.?[0-9]*%?", col_values.iloc[-1]):
                    continue
                else:
                    efficiency_df = efficiency_df.drop(columns=[col_name])

        efficiency_df = efficiency_df.apply(cleanup_and_convert_tofloat)
        efficiency_df = check_empty_dataframe(efficiency_df)
        
        isc_df = df.filter(regex=re.compile(pattern_isc, re.IGNORECASE), axis=1)
        isc_df = isc_df.apply(cleanup_and_convert_tofloat)
        isc_df = check_empty_dataframe(isc_df)

        voc_df = df.filter(regex=re.compile(pattern_voc, re.IGNORECASE), axis=1)
        voc_df = voc_df.apply(cleanup_and_convert_tofloat)
        voc_df = check_empty_dataframe(voc_df)

        impp_df = df.filter(regex=re.compile(pattern_impp, re.IGNORECASE), axis=1)
        impp_df = impp_df.apply(cleanup_and_convert_tofloat)
        impp_df = check_empty_dataframe(impp_df)
        
        vmpp_df = df.filter(regex=re.compile(pattern_vmpp, re.IGNORECASE), axis=1)
        vmpp_df = vmpp_df.apply(cleanup_and_convert_tofloat)
        vmpp_df = check_empty_dataframe(vmpp_df)
        
        pmpp_df = df.filter(regex=re.compile(pattern_pmpp, re.IGNORECASE), axis=1) 
        pmpp_df = pmpp_df.apply(cleanup_and_convert_tofloat)
        pmpp_df = check_empty_dataframe(pmpp_df)
        
        ff_df = df.filter(regex=re.compile(pattern_ff, re.IGNORECASE), axis=1)
        ff_df = ff_df.apply(cleanup_and_convert_tofloat)
        ff_df = check_empty_dataframe(ff_df)
        
        electric_params_stc_df = pd.concat([efficiency_df, isc_df, voc_df, impp_df, vmpp_df, pmpp_df, ff_df], axis="columns")
        
        print(electric_params_stc_df)
        
        #if there's multiple columns with the same name in one excel sheet, merge all the values into one column
        if electric_params_stc_df.columns.is_unique == False:
            uniq = electric_params_stc_df.columns.unique()

            ## Dont know what is going wrong here.
            print("HELLO")
            print(uniq)
            
            electric_params_stc_df = pd.concat([electric_params_stc_df[c].melt()['value'] for c in uniq], axis=1, keys=uniq)
        ##--
        #electric_params_stc_df.apply(cleanup_and_convert_tofloat)
        cols = ['eta / %','ISC / A','VOC / V','IMPP / A','VMPP / V', 'PMPP / W', 'Full factor']
    
        electric_params_stc_df.columns = cols
        # electric_params_stc_df.insert(0, 'filename(.xlsx)', "etst") #change the excel_file_path to excel_file_name later
        # electric_params_stc_df['filename(.xlsx)'] = electric_params_stc_df['filename(.xlsx)'].duplicated().replace({True:'',False:"etst"}) #change the excel_file_path to excel_file_name later, could be more elegant and maybe not necessary?

    
        # print(electric_params_stc_df)
        return electric_params_stc_df

    elif case == "temp_coeff":
        if check_table_title_temp_coeff == False : 

            temp_coeff_voc_df = df.filter(regex = re.compile(temp_coeff_voc, re.IGNORECASE), axis=1)
            temp_coeff_isc_df = df.filter(regex = re.compile(temp_coeff_isc, re.IGNORECASE), axis=1)
            temp_coeff_pmpp_df = df.filter(regex = re.compile(temp_coeff_pmpp, re.IGNORECASE), axis=1)
        
        elif check_table_title_temp_coeff == True : 

            temp_coeff_voc_df = df.filter(regex = re.compile(temp_coeff_voc_ver2, re.IGNORECASE), axis=1)
            temp_coeff_isc_df = df.filter(regex = re.compile(temp_coeff_isc_ver2, re.IGNORECASE), axis=1)
            temp_coeff_pmpp_df = df.filter(regex = re.compile(temp_coeff_pmpp_ver2, re.IGNORECASE), axis=1)
            
            
        temp_coeff_voc_df = check_empty_dataframe(temp_coeff_voc_df)     
        temp_coeff_isc_df = check_empty_dataframe(temp_coeff_isc_df)
        temp_coeff_pmpp_df = check_empty_dataframe(temp_coeff_pmpp_df)

        all_temp_coeff_df = pd.concat([ temp_coeff_voc_df,temp_coeff_isc_df, temp_coeff_pmpp_df], axis="columns")
        all_temp_coeff_df = all_temp_coeff_df.apply(cleanup_and_convert_tofloat)

        cols_temp_coeff = ['T_VOC/%','T_ISC/%','T_PMPP/%']
        all_temp_coeff_df.columns = cols_temp_coeff
        return (all_temp_coeff_df)


def locate_headername_row(df, pattern):
    '''
    -look at the first column of the dataframe and counts how many strings match the predefined pattern
    '''
    headername_row = df.iloc[:, 0].str.contains(pattern, flags=re.IGNORECASE, regex=True, na=False)
    sum_true_row = headername_row.sum() 
    return sum_true_row
    

def locate_headername_column(df, pattern):
    '''
    -iterate over the rows of the dataframe and determine which row contains the headername 
    for the electrical parameters table
    '''
    index = 0
    sum_true_column = 0
    for idx in range(len(df.index)):
            index = idx
            headername_column = df.iloc[idx].str.contains(pattern, flags=re.IGNORECASE, regex=True, na=False)
            sum_true_column = headername_column.sum()

            if sum_true_column > 4: #4 = threshold value
                break
    return sum_true_column, index



def get_file_paths_and_names(dirname):
    all_file_names = []
    all_file_paths = [] #create array as placeholder for all of the file paths
    for root, dirs, files in os.walk(dirname):    # using os.walk to go the lowest "directory?", where all the files are located
        for filename in files:
            if filename.lower().endswith('.xlsx') and filename.lower().startswith('~$') is False:   #check if file extension is ".xlsx" or ".XLSX"
                all_file_paths.append(os.path.join(root, filename))  #append/add each file path to the array
                all_file_names.append(filename)
    all_file_names = [s.replace(".xlsx","") for s in all_file_names]
    return all_file_paths, all_file_names #return the completed array which contains all of the file paths and names


def doall(excel_file_path):

    spreadsheet_file = pd.ExcelFile(excel_file_path)

    worksheets = spreadsheet_file.sheet_names #getting all the sheet names in the excel file

    complete_stc_df = pd.DataFrame(columns=['eta / %','ISC / A','VOC / V','IMPP / A','VMPP / V', 'PMPP / W', 'Full factor'])
    complete_temp_coeff_df = pd.DataFrame(columns=['T_VOC/%','T_ISC/%','T_PMPP/%'])

    check_split_tables = False #
    counter_worksheets = -1


    for sheet_name in worksheets:
        
        active_df = pd.read_excel(spreadsheet_file, sheet_name, header=None)
        

        #remove any whitespaces in the data frame
        active_df = active_df.replace(r'\s+', '', regex=True)

        #check if excel sheet(or dataframe) is empty
        if active_df.empty == True:
            continue

        check_same_table = True
        counter_worksheets += 1
       
        active_df = active_df.dropna(axis=0, how='all') #delete rows if all columns contain NaN - 

        
        if check_split_tables == True:
            check_headername_row_exist = locate_headername_row(active_df,pattern_electrical_params)
            check_headername_column_exist = locate_headername_column(active_df, pattern_electrical_params)
            ##check if the splitted table has headername or not
            if check_headername_row_exist == 0 and check_headername_column_exist[0] == 0:
               previous_table_df =  pd.read_excel(pd.ExcelFile(excel_file_path), worksheets[counter_worksheets-1], header=None)
               num_columns_previous_table = len(previous_table_df.iloc[-1])
               last_row_table_previous_worksheet = previous_table_df.iloc[-1]
               #check if the number of columns in the current df is the same in previous worksheet
               if num_columns_previous_table == len(active_df.iloc[0]):
                  #check whether each column has the same number of characters
                  for column_previous_table, column_current_table in zip(last_row_table_previous_worksheet.astype(str), active_df.iloc[0].astype(str)):
                      difference_in_char_length = abs(len(column_previous_table) - len(column_current_table))
                      if difference_in_char_length > 3:
                          check_same_table = False
                          break
                        
                        #index stc - which row contains the headername
                        #previous_table_df.iloc[index_stc]
                  if check_same_table == True:
                    active_df = active_df.append(previous_table_df.iloc[index_stc], ignore_index=True)
                    active_df = active_df.apply(np.roll, shift=1)

                    active_df = active_df.rename(columns=active_df.iloc[0])
                    table_stc_column = filter_stc_data(active_df, 'electrical_params')
                    complete_stc_df = complete_stc_df.append(table_stc_column)
                    check_split_tables = False
                    continue
                        

        #check if the table title "back efficiency, bifacial, rear gain" exists and if True, ignore it(for now?) and move on to next worksheet
        check_table_title_bifacial = False
        title_bifacial_row = locate_headername_row(active_df, table_title_bifacial)
        title_bifacial_column = locate_headername_column(active_df, table_title_bifacial)
        if title_bifacial_row >=1 or title_bifacial_column[0] >=1:
            check_table_title_bifacial = True
            continue


        ###check if the table title "temperature coefficient" exists
        check_table_title_temp_coeff = False
        title_temp_coeff_row = locate_headername_row(active_df, table_title_temp_coeff)
        title_temp_coeff_column = locate_headername_column(active_df, table_title_temp_coeff)
        if title_temp_coeff_row >=1 or title_temp_coeff_column[0] >=1:
            check_table_title_temp_coeff = True



        ### determine whether header names(Uoc, Pmpp, TkIsc, TkVoc, etc.) exist in the rows or columns
        
        
        if check_table_title_temp_coeff == False:
            true_count_temp_coeff_row = locate_headername_row(active_df, pattern_temp_coeff)
            true_count_temp_coeff_column, index_temp_coeff = locate_headername_column(active_df, pattern_temp_coeff)

        elif check_table_title_temp_coeff == True:

            true_count_temp_coeff_row = locate_headername_row(active_df, pattern_temp_coeff_ver2)
            true_count_temp_coeff_column, index_temp_coeff = locate_headername_column(active_df, pattern_temp_coeff_ver2)
        
        
        true_count_stc_row = locate_headername_row(active_df, pattern_electrical_params)
        true_count_stc_column, index_stc = locate_headername_column(active_df, pattern_electrical_params)
    

        ###

        if true_count_stc_row > 4 and true_count_temp_coeff_column < 3 and true_count_temp_coeff_row < 3:                     #5 = threshold value
            
            active_df = active_df.T #transpose the dataframe
            active_df = active_df.rename(columns=active_df.iloc[0]).drop(active_df.index[0]).reset_index(drop=True).fillna('') #make the zeroth row as the column name and resetting the index
            #print(active_df)
            table_stc_row = filter_stc_data(active_df, 'electrical_params', check_table_title_temp_coeff)
            complete_stc_df = complete_stc_df.append(table_stc_row)

            #check_split_tables == True

        elif true_count_stc_column > 4 and true_count_temp_coeff_column < 3 and true_count_temp_coeff_row < 3: #5 = threshold value

            active_df = active_df.rename(columns=active_df.iloc[index_stc])
            table_stc_column = filter_stc_data(active_df, 'electrical_params', check_table_title_temp_coeff)
            complete_stc_df = complete_stc_df.append(table_stc_column)

            check_split_tables = True
            
        elif true_count_temp_coeff_column >2 and true_count_stc_column < 5 and true_count_stc_row < 5: #3 = threshold value

             #check whether the values of temp. coeffs and the string 'temperature coefficient' is in one row


            active_df = active_df.rename(columns=active_df.iloc[index_temp_coeff])
            table_temp_coeff_column = filter_stc_data(active_df, 'temp_coeff', check_table_title_temp_coeff)
            complete_temp_coeff_df = complete_temp_coeff_df.append(table_temp_coeff_column)

        elif true_count_temp_coeff_row > 2 and true_count_stc_column < 5 and true_count_stc_row < 5: #3 = threshold value  

            active_df = active_df.T 

            #need to check whether first row contains any number
            check_digits = '[+-]\d\.\d+|[+-][0-9]\.[0-9]+'
            if active_df.iloc[0].str.contains(check_digits, flags=re.IGNORECASE, regex=True, na=False).any():
                active_df = active_df.rename(columns=active_df.iloc[0]).reset_index(drop=True).fillna('')
                table_temp_coeff_row = filter_stc_data(active_df, 'temp_coeff', check_table_title_temp_coeff)
            else:
                active_df = active_df.rename(columns=active_df.iloc[0]).drop(active_df.index[0]).reset_index(drop=True).fillna('') #make the zeroth row as the column name and resetting the index
                table_temp_coeff_row = filter_stc_data(active_df, 'temp_coeff', check_table_title_temp_coeff)
            complete_temp_coeff_df = complete_temp_coeff_df.append(table_temp_coeff_row)

        #case for when the electrical parameters table and temp. coeff table is in one table
        #for now, still ignoring the temp. coeffs data otherwise ERRORS :(
        elif true_count_stc_row > 4 and check_table_title_temp_coeff==True:
            
            active_df = active_df.drop(index_temp_coeff)
            active_df = active_df.T #transpose the dataframe
            active_df = active_df.rename(columns=active_df.iloc[0]).drop(active_df.index[0]).reset_index(drop=True).fillna('') #make the zeroth row as the column name and resetting the index
            #print(active_df)
            table_stc_row = filter_stc_data(active_df, 'electrical_params', check_table_title_temp_coeff)
            complete_stc_df = complete_stc_df.append(table_stc_row)

            

    complete_stc_df = complete_stc_df.reset_index(drop=True)
    complete_temp_coeff_df = complete_temp_coeff_df.reset_index(drop=True)
    complete_dataframe = pd.concat([complete_stc_df, complete_temp_coeff_df], axis=1)
    complete_dataframe[['T_VOC/%','T_ISC/%','T_PMPP/%']] = complete_dataframe[['T_VOC/%','T_ISC/%','T_PMPP/%']].fillna(method='ffill') #propagate[s] last valid observation forward to next valid (filling nan with preceding values)

    return complete_stc_df, complete_temp_coeff_df





###                                    ------   MAIN ------       ###
### !!!IMPORTANT!!! : ALL EXCEL FILES IN THE DESIGNATION FOLDER NEED TO BE CLOSED FIRST, OTHERWISE ERROR @PD.READ_EXCEL -solved on macOS, test needed on Windows
### !!!IMPORTANT!!! : OUTPUT EXCEL FILE WILL BE LOCATED IN THE SAME DIRECTORY WHERE THIS .PY CODE IS LOCATED


#patterns to detect if the table in the active worksheet is containing temperature coefficients
temp_coeff_isc = 'tk(current|isc)|(short\-?circuit)?\-?currenttemperaturecoefficient-?(isc)?|isc\.?temp\.?coef\.?|current[\[\(]?(%\/K|alpha)[\]\)]?'
temp_coeff_voc = 'tk(voltage|[uv]oc)|(open\-?circuit)?\-?voltagetemperaturecoefficient-?([vu]oc)?|[vu]oc\.?temp\.?coef\.?|voltage[\[\(]?(%\/K|beta)[\]\)]?'
temp_coeff_pmpp = 'tk(power|pmax)|(max\.?-?power|power)temperaturecoefficient(of)?(pmpp|pmax)?|^temperaturecoefficientof(pmpp|pmax)|(pm|pmax)\.?temp\.?coef\.?|power[\[\(]?(%\/K|gamma)[\]\)]?'
pattern_temp_coeff = "(%s|%s|%s)" % (temp_coeff_isc, temp_coeff_voc, temp_coeff_pmpp)

##added additional patterns IF the table title "temperature coefficients" is confirmed to exist in the current worksheet
table_title_temp_coeff = '^temp(\.|erature)?(index|[ck]oeffi[cz]ient[s]?|characteristic[s]?):*'
temp_coeff_isc_ver2 = temp_coeff_isc + '|⍺|current|isc'
temp_coeff_voc_ver2 = temp_coeff_voc +  '|voc|ß|voltage'
temp_coeff_pmpp_ver2 = temp_coeff_pmpp + '|power|pmax|γ|δ|pm'
pattern_temp_coeff_ver2 = "(%s|%s|%s)" % (temp_coeff_isc_ver2, temp_coeff_voc_ver2, temp_coeff_pmpp_ver2)

#patterns to detect the electrical parameters
pattern_efficiency = "eff((?!code).)*$|ncell|model\(%\)"
pattern_isc = "isc|shortcircuit(current)?"
pattern_voc = "[uv]oc|opencircuit(voltage)?|vm\W|circuitvoltage"
pattern_impp = "^imp+|^ip+m|(max\.?(imum)?)?powercurrent|currentat\s(max\.?(imum)?)?power|im(?!um)$"
pattern_vmpp = "^[uv]mp+|^[uv]p+m|(max\.?(imum)?)?powervoltage|voltageat\s(max\.?(imum)?)?power"
pattern_pmpp = "pmax|pmpp|ppm|^pm|^power$|[\(\[]wp*[\)\]]|(average|rated|charged)power"
pattern_ff = "^ff|fillfactor"

#patterns used for detecting if the table title "bifacial, rear, back efficiency, etc." exists
table_title_bifacial = 'back(electrical|efficiency)?|rear|bifacial'

##patterns used for detecting if the table in the active worksheet is containing electrical parameters@STC
pattern_electrical_params = pattern_efficiency + "|" + pattern_isc + "|" + pattern_voc + "|" + pattern_impp + "|" + pattern_vmpp + "|" + pattern_pmpp + "|" + pattern_ff


def main():

    start_time = time.time()



    excel_folder_path = '/Volumes/T7/thesis-data/test/single_excel/'
    all_excel_file_paths, all_excel_file_names = get_file_paths_and_names(excel_folder_path)
    output_dataframe = list()



    ## This is the main loop



    for excel_file_path, excel_file_name in zip(all_excel_file_paths, all_excel_file_names): 
        #print(excel_file_path)
        #print(excel_file_name)

        #spreadsheet_file = pd.ExcelFile(excel_file_path)
        #excel_file_path = '/Users/benedictus/Documents/Data_Mining_pytorch/pandas_test_cell_real.xlsx'
        
        complete_dataframe = doall(excel_file_path)
        output_dataframe.append(complete_dataframe)
            

    #output_stc_df = pd.concat(complete_stc_df, ignore_index=True)
    end_time = time.time()
    time_lapsed = end_time - start_time
    time_convert(time_lapsed)

    print('----TEST END----')

    #create an output excel file that contains all the extracted dataframe
    output_dataframe = pd.concat(output_dataframe, ignore_index=True)
    output_dataframe.to_excel("output_test.xlsx",index=False)

if __name__=="__main__":
    main()

#####


