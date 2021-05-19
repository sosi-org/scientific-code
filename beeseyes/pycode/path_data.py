import numpy as np

from openpyxl import load_workbook

def load_xls(filename_xls, columnset):
    wb = load_workbook(filename_xls, read_only=True)
    #print (wb.sheetnames)
    #ws = wb.get_sheet_by_name('Sheet1')
    ws = wb.active

    ret = {}
    #use_col = 0
    for key in columnset:
        # key =- new header
        #columns: tuple = columnset[key]
        columns = columnset[key]

        print('columns', columns)

        res = []
        #for use_col in columns:
        for use_col in columns:
            print('column', use_col, 'of', key)
            #rows = ws.iter_rows()
            #header = ws.iter_rows().value
            for row in ws.iter_rows(min_row=1):
                header = row[use_col].value
                continue
            x2 = np.array([ float(r[use_col].value) for r in ws.iter_rows(min_row=2)])
            #x2 = np.array([float(r[use_col].value) for r in ws.iter_rows(*)])

            header = [r[use_col].value for r in ws.iter_rows(min_row=1)][0]

            #return x2[1:], x2[0]
            #res.append((x2, header))
            print('header',header,'for column', use_col);
            res.append(x2)
            print('', flush=True)
        res2d = np.array(res).T
        assert len(res2d.shape) == 2
        assert res2d.shape[1] == len(columns)
        print('res2d.shape', res2d.shape)
        ret[key] = res2d

    return ret

def load_trajectory_data(filename_xls):
    cols = {
     'fTime': (0,),
     'RWSmoothed': (6,7,8), # RWxSmoothed
     'direction': (10,11,12),
    }
    #x2, label = load_xls(cols)[0]
    #print('header:', label)
    allv = load_xls(filename_xls, cols)
    print('allv:', allv)
    print(allv)
    return allv

if __name__ == "__main__":
    CURRENT_PATH = '/Users/a9858770/cs/scientific-code/beeseyes'
    POSITIONS_XLS = CURRENT_PATH + '/Setup/beepath.xlsx'

    load_trajectory_data(POSITIONS_XLS)
