
class DefaultSetting:
    def __init__(self) -> None:
        db_name = None
        db_dir = None
        table_name_cols_map = dict()
        label_names = list()

class EICUDefaultSettings(DefaultSetting):
    def __init__(self, db_path) -> None:
        super().__init__()
        self.db_name = "eicu"
        self.db_dir = db_path
        self.table_name_cols_map = {
            'patient': ['uniquepid', 
                    'patienthealthsystemstayid', 
                    'patientunitstayid', 
                    'hospitaladmitoffset',
                    'hospitaldischargeoffset',      # Endtime of a hospital.
                    'gender',
                    'age',
                    'admissionweight',
                    'admissionheight',
                    'unitdischargestatus'           # Whether a patient dies.
                    ],

            'diagnosis': ['patientunitstayid', 
                            'diagnosisoffset', 
                            'icd9code'],

            'medication': ['patientunitstayid',
                                'drugname',
                                'dosage',
                                'drugorderoffset'],
            
            'lab': ['patientunitstayid',
                        'labname',
                        'labresult',
                        'labresultoffset']
        }        
        self.label_names = ["518", "584", "599", "575", "428", "Normal", "Expired", "END"]
