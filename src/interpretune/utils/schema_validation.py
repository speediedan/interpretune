from jsonschema import Draft202012Validator, validators

# # Example usage:
# obj = {'foo': 'bar', 'not_in_schema': 'nope not this'}
# schema = {'properties': {'foo': {'type': 'string'}}}
#
# RemoveAdditionalPropertiesValidator(schema).validate(obj)
# print(obj)
# # Expected output: {'foo': 'bar'}


def extend_validator(validator_class):
    validate_properties = validator_class.VALIDATORS["properties"]

    def remove_additional_properties(validator, properties, instance, schema):
        for prop in list(instance.keys()):
            if prop not in properties:
                del instance[prop]
        for error in validate_properties(validator, properties, instance, schema):
            yield error
        return

    return validators.extend(validator_class, {"properties": remove_additional_properties})


RemoveAdditionalPropertiesValidator = extend_validator(Draft202012Validator)
