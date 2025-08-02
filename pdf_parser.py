import pymupdf
import re
import os
import streamlit as st

class read_library_search:
    def __init__(self,file):
        self.file = file
    
    def read_library_search(self):
        list_of_pg_w_uv = []
        chosen_library_search = self.file
        doc = pymupdf.open(chosen_library_search)
        text=""
        for page in doc:
            text += page.get_text()
            #Check if page contains UV Spectrum:
            if "Target + Library Spectrum" in page.get_text() or "Print of window" in page.get_text():
                list_of_pg_w_uv.append(page.number)
            else:
                st.write("No UV Spectrum detected.")

        # Library Search starts 
        pattern = r"-{7}\|-{7}\|-{7}\|---\|-{10}\|-{6}\|-\|-{6}\|-{20}\n(.*?)Data File"

        try:
            matches = re.findall(pattern, text, re.DOTALL)
            
            # Library search will end with "Note(s):"
            combined_matches = ''.join(matches).split("Note(s):")[0]
            
            #combined_matches = combined_matches.split("\n").split("    ")[0]

            list_combined_matches = combined_matches.split("\n")

            list_combined_matches = [b for b in list_combined_matches if b]

            test = [i.split(" ") for i in list_combined_matches]

            test = [[x for x in y if x.strip() and x!="u" and x!="d"] for y in test]

            compiled_list = []
            for listmini in test:
                #compiled_dict = {}
                if abs(float(listmini[0]) - float(listmini[1])) <= 1 and int(listmini[7])>=950:
                    compiled_list.append(" ".join(listmini[8:]))

            combined_compiled_list = ("$").join(compiled_list)
        except Exception as e:
            st.write("File was not library search.")

        # if len(list_of_pg_w_uv) != 0 :
        #     #loop through uv spectrums
        #     for page_no in list_of_pg_w_uv:
        #         if doc[page_no] 

        return (combined_compiled_list)
    
    def fill_in_form(self):
        file_path="Pseudo Control Form.pdf"
        
        header = ['Effect Grouping', 'All adulterants: Not Detected', 'All other adulterants: Not Detected', 'Detected Adulterants']        
        doc = pymupdf.open(file_path)
        
        dict_for_agent = {}
        for page_num, page in enumerate(doc):
            widgets = page.widgets()  # List of form fields (widgets) on the page
            list_of_widgets= []
            if widgets:
                for widget in widgets:
                    field_name = widget.field_name
                    list_of_widgets.append(field_name)
            tabs = page.find_tables()
            if tabs.tables:
                table_print = tabs[0].extract()
            count = 0
            for row in table_print:
                for i in range(len(row)):
                    if row[i] == "":
                        row[i] = list_of_widgets[count]
                        count+=1
                    else:
                        row[i] = row[i].replace("\n"," ")
                dict_for_parsing_form_fields = {}
                dict_for_parsing_form_fields[f"{row[0]} which is a category of {header[0]}"] = [f"Textfield {row[i]} is a field in the pdf that represents {header[i]}" for i in range(1,len(row))]

        doc.close()

        return dict_for_parsing_form_fields
