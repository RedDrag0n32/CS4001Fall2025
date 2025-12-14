# save_I_E.py
I_E = input("Enter the current injection amplitude (I_E): ")

# Save to a file
with open("I_E_Value.txt", "w") as f:
    f.write(str(I_E))

print("I_E value saved to I_E_Value.txt")



