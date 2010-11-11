local File = torch.getmetatable('torch.File')

function File:writeBool(value)
   if value then
      self:writeInt(1)
   else
      self:writeInt(0)
   end
end

function File:readBool()
   return (self:readInt() == 1)
end

local TYPE_NIL     = 0
local TYPE_NUMBER  = 1
local TYPE_STRING  = 2
local TYPE_TABLE   = 3
local TYPE_TORCH   = 4
local TYPE_BOOLEAN = 5

function File:writeObject(object)
   -- we use an environment to keep a record of written objects
   if not torch.getenv(self).writeObjects then
      torch.setenv(self, {writeObjects={}, writeObjectsRef={}, readObjects={}})
   end

   -- if nil object, only write the type and return
   if type(object) ~= 'boolean' and not object then
      self:writeInt(TYPE_NIL)
      return
   end

   -- check the type we are dealing with
   local type = type(object)
   if torch.typename(object) then
      type = TYPE_TORCH
   elseif type == 'table' then
      type = TYPE_TABLE
   elseif type == 'number' then
      type = TYPE_NUMBER
   elseif type == 'string' then
      type = TYPE_STRING
   elseif type == 'boolean' then
      type = TYPE_BOOLEAN
   else
      error('unwritable object')
   end
   self:writeInt(type)

   if type == TYPE_NUMBER then
      self:writeDouble(object)
   elseif type == TYPE_BOOLEAN then
      self:writeBool(object)
   elseif type == TYPE_STRING then
      local stringStorage = torch.CharStorage():string(object)
      self:writeInt(#stringStorage)
      self:writeChar(stringStorage)
   elseif type == TYPE_TORCH or type == TYPE_TABLE then
      -- check it exists already (we look at the pointer!)
      local objects = torch.getenv(self).writeObjects
      local objectsRef = torch.getenv(self).writeObjectsRef
      local index = objects[torch.pointer(object)]

      if index then
         -- if already exists, write only its index
         self:writeInt(index)
      else
         -- else write the object itself
         index = objects.nWriteObject or 0
         index = index + 1
         objects[torch.pointer(object)] = index
         objectsRef[object] = index -- we make sure the object is not going to disappear
         self:writeInt(index)
         objects.nWriteObject = index

         if type == TYPE_TORCH then
            local version   = torch.CharStorage():string('V ' .. torch.version(object))
            local className = torch.CharStorage():string(torch.typename(object))
            if not torch.factory(torch.typename(object)) then
               error(torch.typename(object) .. ' is a non-serializable Torch object')
            end
            self:writeInt(#version)
            self:writeChar(version)
            self:writeInt(#className)
            self:writeChar(className)
            object:write(self)
         else -- it is a table
            local size = 0; for k,v in pairs(object) do size = size + 1 end
            self:writeInt(size)
            for k,v in pairs(object) do
               self:writeObject(k)
               self:writeObject(v)
            end
         end
      end
   else
      error('unwritable object')
   end
end

function File:readObject()
   -- we use an environment to keep a record of read objects
   if not torch.getenv(self).writeObjects then
      torch.setenv(self, {writeObjects={}, writeObjectsRef={}, readObjects={}})
   end

   -- read the type
   local type = self:readInt()

   -- is it nil?
   if type == TYPE_NIL then
      return nil
   end

   if type == TYPE_NUMBER then
      return self:readDouble()
   elseif type == TYPE_BOOLEAN then
      return self:readBool()
   elseif type == TYPE_STRING then
      local size = self:readInt()
      return self:readChar(size):string()
   elseif type == TYPE_TABLE or type == TYPE_TORCH then
      -- read the index
      local index = self:readInt()

      -- check it is loaded already
      local objects = torch.getenv(self).readObjects
      if objects[index] then
         return objects[index]
      end

      -- otherwise read it
      if type == TYPE_TORCH then
         local version, className, versionNumber
         version = self:readChar(self:readInt()):string()
         versionNumber = tonumber(string.match(version, '^V (.*)$'))
         if not versionNumber then
            className = version
            versionNumber = 0 -- file created before existence of versioning system
         else
            className = self:readChar(self:readInt()):string()
         end
         if not torch.factory(className) then
            error('unknown Torch class ' .. className)
         end
         local object = torch.factory(className)()
         objects[index] = object
         object:read(self, versionNumber)
         return object
      else -- it is a table
         local size = self:readInt()
         local object = {}
         objects[index] = object
         for i = 1,size do
            local k = self:readObject()
            local v = self:readObject()
            object[k] = v
         end
         return object
      end
   else
      error('unknown object')
   end
end
