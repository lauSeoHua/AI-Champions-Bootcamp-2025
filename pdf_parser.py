import pymupdf
import re
import os
import streamlit as st
import fitz
from io import BytesIO

class read_library_search:
    def __init__(self,file):
        self.file = file
    
    def read_library_search(self):
        with st.spinner("📄🔍Opening PDF and reading it📄🔍..."):
            list_of_pg_w_uv = []
            chosen_library_search = self.file
            doc = fitz.open(stream=chosen_library_search.read(), filetype="pdf")
            text=""
            for page in doc:
                
                #Check if page contains UV Spectrum:
                if "Target + Library Spectrum" in page.get_text() or "Print of window" in page.get_text():
                    list_of_pg_w_uv.append(page.number)
                else:
                    text += page.get_text()
                    
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
    
    def fill_in_form(self,list_of_cpds):
        with st.spinner("📄🔍 Getting PDF control forms fields📄🔍..."):
            file_path="Pseudo Control Form.pdf"
            
            header = ['Effect Grouping', 'All adulterants: Not Detected', 'All other adulterants: Not Detected', 'Detected Adulterants']        
            doc = fitz.open(file_path)
            dict_for_parsing_form_fields = {}
            
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
                    
                    dict_for_parsing_form_fields[row[0]] = [row[i] for i in range(1,len(row))]
                    
            # Fill in start now
            undetected_field = []
            for key,value in dict_for_parsing_form_fields.items():
                undetected_field.append(value[0])
            #list_to_add : ["Doxycycline$Anti-biotics (Acne),Anti-biotics (Internal Use)","Oseltamivir$Anti-virals "]
            for cpds in list_of_cpds:
                cpd = cpds.split("$")[0]
                grps = cpds.split("$")[1]
                if "," in grps:
                        grp_list = grps.split(",")
                else:
                    grp_list = [grps]
            
                for grp in grp_list:
                    grp = grp.strip()
                    field_names = dict_for_parsing_form_fields.get(grp)
                    if field_names:
                        for page_num, page in enumerate(doc):
                            widgets = page.widgets()

                            field_to_update = field_names[2]
                            yes_field = field_names[1]
                            undetected = field_names[0]

                            for widget in widgets:
                                if widget.field_name == field_to_update:
                                    curr_word = widget.field_value or ""
                                    new_value = curr_word + "," + cpd if curr_word else cpd
                                    
                                    widget.field_value = new_value
                                    widget.update()

                                elif widget.field_name == yes_field:
                                    widget.field_value = "Yes"
                                    widget.update()

                            if undetected in undetected_field:
                                undetected_field.remove(undetected)

            for page_num, page in enumerate(doc):
                widgets = page.widgets()                    

                for widget in widgets:
                    for undetected in undetected_field:
                        if widget.field_name == undetected:
                            widget.field_value = "Yes"
                            widget.update()
            # Save to a BytesIO object
            pdf_bytes = BytesIO()
            doc.save(pdf_bytes)
            doc.close()

            # Move pointer to the start
            pdf_bytes.seek(0)
            st.download_button(
            label="📥 Download Filled PDF",
            data=pdf_bytes,
            file_name="filled_form.pdf",
            mime="application/pdf"
        )
